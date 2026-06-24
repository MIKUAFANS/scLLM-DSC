import os
import threading
from typing import Tuple
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm
from dotenv import load_dotenv
from joblib import Parallel, delayed

from openai import OpenAI
from .utils import embedding_api

load_dotenv()

def check_gene_in_ncbi(ncbi_dataset_gene, gene_names):
    count = 0
    cell_list = []
    for gene in gene_names:
        flag = ncbi_dataset_gene.loc[ncbi_dataset_gene["gene_name"].str.lower() == gene.lower(), "GeneID"]
        if flag.empty or flag.iloc[0] == "NA":
            count += 1
            cell_list.append(gene)
            
    return count, cell_list

def split_cell_by_length(cell, max_length=8000):
    """
    如果 cell 字符串的长度超过 max_length，则按逗号分割，确保每个子字符串长度不超过 max_length。
    """
    # 如果字符串长度小于最大长度，不需要分割，直接返回
    if len(cell) <= max_length:
        return [cell]

    # 否则按逗号分割
    cell_parts = cell.split(",")
    current_part = ""
    split_parts = []

    # 按最大长度拆分
    for part in cell_parts:
        # 如果当前部分 + 新部分超过最大长度，则保存当前部分，开始新的一部分
        if len(current_part) + len(part) + 1 > max_length:  # +1 是为了加入逗号
            split_parts.append(current_part)
            current_part = part
        else:
            if current_part:
                current_part += "," + part
            else:
                current_part = part

    # 添加最后的部分
    if current_part:
        split_parts.append(current_part)

    return split_parts

def cell_embeddings(cell: list, openai_client: OpenAI, model:str="text-embedding-3-small", batch_size:int=100, max_length:int=8000) -> np.ndarray:
    openai_client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
    # embeddings = []
    # for cell in tqdm(cell_desc, desc="Generating cell embeddings"):
    cells = split_cell_by_length(cell, max_length=max_length)
    # cells = cell.split(",")
    cell_embedding = embedding_api(openai_client, cells, model=model, batch_size=batch_size, timeout=2)
    
    return np.mean(cell_embedding, axis=0)
        # embeddings.append(np.mean(cell_embedding, axis=0))

    # return np.array(embeddings)
    
def process_cells_parallel(cells, openai_client: OpenAI, max_length=8000, batch_size=100, n_jobs=-1) -> np.ndarray:
    return Parallel(n_jobs=n_jobs)(
        delayed(cell_embeddings)(cell, openai_client, model="text-embedding-3-small", batch_size=batch_size, max_length=max_length)
        for cell in tqdm(cells)
    )
    # embeddings = []
    # for cell in tqdm(cells, desc="Generating cell embeddings in parallel"):
    #     embeddings.append(cell_embeddings(cell, openai_client, model="text-embedding-3-small", batch_size=batch_size, max_length=max_length))

class GeneSelector:
    def __init__(self, filename: str, save_path: str, n_jobs: int = -1):
        self.filename = filename
        self.openai_client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
        self.ncbi_dataset_gene = pd.read_csv(os.path.join(save_path, "gene_descriptions_with_embeddings.csv"))
        self.n_jobs = n_jobs
        self._ncbi_lock = threading.RLock()

    def load_dataset(self):
        if self.filename.endswith(".h5") or self.filename.endswith(".hdf5"):
            with h5py.File(self.filename, "r") as f:
                try:
                    # print( f.keys())
                    if "gene_names" in f:
                        gene_names = f["gene_names"][...]
                    elif "var_names" in f:
                        gene_names = f["var_names"][...]
                    else:
                        gene_names = f["gene_name"][...]
                        
                    if "X" in f:
                        X = f["X"][...]
                    elif "data" in f:
                        X = f["data"][...]
                    elif "exprs" in f:
                        exprs_handle = f["exprs"]
                        X = sp.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                                exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
                    
                    X = X.astype('float32')
                except KeyError:
                    print(f"Keys available in the dataset: {list(f.keys())}")
                    raise KeyError(f"Keys available in the dataset: {list(f.keys())}")
                
                adata = sc.AnnData(X=X, var={"gene_names": gene_names.astype(str)})
        elif self.filename.endswith(".h5ad"):
            adata = sc.read_h5ad(self.filename)
            print()
            adata.var["gene_names"] = adata.var["feature_name"].astype(str)

        return adata
    
    # def check_gene_in_ncbi(self, gene_names):
    #     count = 0
    #     cell_list = []
    #     for gene in gene_names:
    #         flag = self.ncbi_dataset_gene.loc[self.ncbi_dataset_gene["gene_name"].str.lower() == gene.lower(), "GeneID"]
    #         if flag.empty or flag.iloc[0] == "NA":
    #             count += 1
    #             cell_list.append(gene)
    #     return count, cell_list
         
    def select_genes(self, adata, top_k=2048, n_jobs=-1):
        """
        并行按 cell 选取 top_k 高表达基因，并生成：
        - cell_strings: 每个细胞一行 "gene:expr,gene:expr,..."
        - unknown_gene_count: 每个细胞未知基因统计
        n_jobs: joblib 并行进程数，默认用所有核。
        """
        n_cells, n_genes = adata.X.shape
        print(f"The dataset has {n_cells} cells and {n_genes} genes.")
        gene_names = adata.var["gene_names"].values

        # -------- 稀疏矩阵情况 --------
        if sp.issparse(adata.X):
            X = adata.X.tocsr()

            def process_cell_sparse(i: int, ncbi_dataset_gene: pd.DataFrame = self.ncbi_dataset_gene):
                row_start, row_end = X.indptr[i], X.indptr[i + 1]
                row_data = X.data[row_start:row_end]
                row_cols = X.indices[row_start:row_end]

                # 只保留表达 > 0 的基因
                mask = row_data > 0
                row_data = row_data[mask]
                row_cols = row_cols[mask]

                if row_data.size == 0:
                    # 没有表达基因
                    cell_str = "None"
                    unknown_count = 0
                    unknown_genes_str = ""
                    return cell_str, unknown_count, unknown_genes_str

                k_i = min(top_k, row_data.size)
                # 找到表达量最大的 k_i 个
                top_idx_part = np.argpartition(row_data, -k_i)[-k_i:]
                top_vals = row_data[top_idx_part]
                top_cols = row_cols[top_idx_part]

                # 按表达量从大到小排序
                order = np.argsort(-top_vals)
                top_vals = top_vals[order]
                top_cols = top_cols[order]

                genes = gene_names[top_cols]
                unknown_count, unknown_genes = check_gene_in_ncbi(ncbi_dataset_gene, genes)
                unknown_genes_str = ", ".join(unknown_genes)

                parts = [f"{g}:{v:.4g}" for g, v in zip(genes, top_vals)]
                cell_str = ",".join(parts)

                return cell_str, unknown_count, unknown_genes_str

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_cell_sparse)(i)
                for i in tqdm(range(n_cells), desc="Building per-cell top2048 strings (sparse)")
            )

        # -------- 稠密矩阵情况 --------
        else:
            X = np.asarray(adata.X)

            def process_cell_dense(i: int, ncbi_dataset_gene: pd.DataFrame = self.ncbi_dataset_gene):
                row = X[i]
                nz_idx = np.where(row > 0)[0]
                if nz_idx.size == 0:
                    cell_str = "None"
                    unknown_count = 0
                    unknown_genes_str = ""
                    return cell_str, unknown_count, unknown_genes_str

                vals = row[nz_idx]
                k_i = min(top_k, nz_idx.size)

                top_idx_part = np.argpartition(vals, -k_i)[-k_i:]
                top_vals = vals[top_idx_part]
                top_cols = nz_idx[top_idx_part]

                order = np.argsort(-top_vals)
                top_vals = top_vals[order]
                top_cols = top_cols[order]

                genes = gene_names[top_cols]
                unknown_count, unknown_genes = check_gene_in_ncbi(ncbi_dataset_gene, genes)
                unknown_genes_str = ", ".join(unknown_genes)

                parts = [f"{g}:{v:.4g}" for g, v in zip(genes, top_vals)]
                cell_str = ",".join(parts)

                return cell_str, unknown_count, unknown_genes_str

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_cell_dense)(i)
                for i in tqdm(range(n_cells), desc="Building per-cell top2048 strings (dense)")
            )

        # -------- 汇总结果到 DataFrame --------
        cell_strings_list = [r[0] for r in results]
        unknown_counts = [r[1] for r in results]
        unknown_genes_strs = [r[2] for r in results]

        cell_strings = pd.DataFrame({"CellString": cell_strings_list})
        unknown_gene_count = pd.DataFrame(
            {
                "CellIndex": np.arange(n_cells),
                "UnknownGeneCount": unknown_counts,
                "UnknownGenes": unknown_genes_strs,
            }
        )

        return cell_strings, unknown_gene_count
    
    # def split_cell_by_length(self, cell, max_length=8000):
    #     """
    #     如果 cell 字符串的长度超过 max_length，则按逗号分割，确保每个子字符串长度不超过 max_length。
    #     """
    #     # 如果字符串长度小于最大长度，不需要分割，直接返回
    #     if len(cell) <= max_length:
    #         return [cell]

    #     # 否则按逗号分割
    #     cell_parts = cell.split(",")
    #     current_part = ""
    #     split_parts = []

    #     # 按最大长度拆分
    #     for part in cell_parts:
    #         # 如果当前部分 + 新部分超过最大长度，则保存当前部分，开始新的一部分
    #         if len(current_part) + len(part) + 1 > max_length:  # +1 是为了加入逗号
    #             split_parts.append(current_part)
    #             current_part = part
    #         else:
    #             if current_part:
    #                 current_part += "," + part
    #             else:
    #                 current_part = part

    #     # 添加最后的部分
    #     if current_part:
    #         split_parts.append(current_part)

    #     return split_parts

    # def cell_embeddings(self, cell_desc: list) -> np.ndarray:
    # def cell_embeddings(self, cell: list, openai_client: OpenAI, model:str="text-embedding-3-small", batch_size:int=100, max_length:int=8000) -> np.ndarray:
    #     # embeddings = []
    #     # for cell in tqdm(cell_desc, desc="Generating cell embeddings"):
    #     cells = self.split_cell_by_length(cell, max_length=max_length)
    #     # cells = cell.split(",")
    #     cell_embedding = embedding_api(openai_client, cells, model=model, batch_size=batch_size)
        
    #     return np.mean(cell_embedding, axis=0)
    #         # embeddings.append(np.mean(cell_embedding, axis=0))

    #     # return np.array(embeddings)
        
    # def process_cells_parallel(self, cells, max_length=8000, batch_size=100, n_jobs=-1) -> np.ndarray:
    #     return Parallel(n_jobs=n_jobs)(
    #         delayed(cell_embeddings)(cell, self.openai_client, model="text-embedding-3-small", batch_size=batch_size, max_length=max_length)
    #         for cell in cells
    #     )
    
    def __call__(self, save_path: str, top_k=2048) -> Tuple[np.ndarray, pd.DataFrame]:
        cell_strings_path = os.path.join(save_path, "cell_top_genes.csv")
        unknown_gene_save_path = os.path.join(save_path, "unknown_gene_counts.csv")
        if os.path.exists(cell_strings_path):
            print(f"Cell top genes file already exists at {cell_strings_path}. Loading existing file.")
            cell_strings = pd.read_csv(cell_strings_path)
            unknown_gene_count = pd.read_csv(unknown_gene_save_path)
        else:
            adata = self.load_dataset()
            cell_strings, unknown_gene_count = self.select_genes(adata, top_k=top_k, n_jobs=self.n_jobs)
            cell_strings.to_csv(cell_strings_path, index=False)
            unknown_gene_count.to_csv(unknown_gene_save_path, index=False)
            print(f"Cell top genes saved to {cell_strings_path}")
            
        embeddings_save_path = os.path.join(save_path, "cell_top_genes_embeddings.npz")
        if os.path.exists(embeddings_save_path):
            print(f"Embeddings file already exists at {embeddings_save_path}. Skipping embedding generation.")
            embeddings = np.load(embeddings_save_path)['embeddings']
        else:
            cell_desc = cell_strings['CellString'].tolist()
            # embeddings = self.cell_embeddings(cell_desc)
            embeddings = process_cells_parallel(cell_desc, None, max_length=8000, batch_size=100, n_jobs=self.n_jobs)
            
            np.savez_compressed(embeddings_save_path, embeddings=embeddings)
            print(f"Embeddings saved to {embeddings_save_path}")
            
        return np.array(embeddings), unknown_gene_count

class GeneEmbeddingWeighter:
    def __init__(self, save_path: str, n_jobs: int = -1):
        self.save_path = save_path
        self.n_jobs = n_jobs  # 并行进程数，默认用完所有核

    def __call__(self, *args, **kwds) -> np.ndarray:
        weighted_cell_embeddings_save_path = os.path.join(
            self.save_path, "weighted_cell_embeddings.npz"
        )
        if os.path.exists(weighted_cell_embeddings_save_path):
            print(
                f"Weighted cell embeddings file already exists at {weighted_cell_embeddings_save_path}. Skipping generation."
            )
            cell_embeddings = np.load(weighted_cell_embeddings_save_path)["embeddings"]
        else:
            # 读入细胞字符串
            cell_strings_path = os.path.join(self.save_path, "cell_top_genes.csv")
            cell_strings = pd.read_csv(cell_strings_path)

            # 读入基因描述 embedding
            gene_desc_embeddings_path = os.path.join(
                self.save_path, "gene_desc_embeddings.npz"
            )
            gene_desc_embeddings = np.load(gene_desc_embeddings_path)["embeddings"]

            # 读入基因描述表（GeneID -> embedding 下标）
            gene_desc = pd.read_csv(os.path.join(self.save_path, "gene_desc.csv"))
            gene_desc["GeneID"] = gene_desc["GeneID"].astype(int)
            # 建立 GeneID -> desc_index 的映射
            gene_desc = gene_desc.reset_index(drop=False)
            # index 列就是 embedding 对应的行号
            id_to_index = dict(
                zip(gene_desc["GeneID"].astype(int).tolist(),
                    gene_desc["index"].astype(int).tolist())
            )

            # 读入 gene_name -> GeneID 映射
            gene_names_with_ids = pd.read_csv(
                os.path.join(self.save_path, "gene_descriptions_with_embeddings.csv")
            )
            gene_names_with_ids["GeneID"] = gene_names_with_ids["GeneID"].fillna(-1)
            gene_names_with_ids["GeneID"] = gene_names_with_ids["GeneID"].astype(int)

            # 建立 gene_name(lower) -> GeneID 的 dict，便于在 worker 里快速查
            name_to_geneid = {}
            for _, row in gene_names_with_ids.iterrows():
                gname = str(row["gene_name"]).lower()
                gid = int(row["GeneID"])
                if gid != -1:
                    name_to_geneid[gname] = gid

            # 单个 cell 的处理逻辑，供并行调用
            def process_cell(cell_string: str) -> np.ndarray:
                gene_entries = cell_string.split(",")
                cell_embedding = np.zeros((gene_desc_embeddings.shape[1],), dtype=np.float32)

                for entry in gene_entries:
                    if not entry:
                        continue
                    # gene_name:expr_value
                    try:
                        gene_name, expr_value = entry.split(":")
                    except ValueError:
                        # 格式不对的就跳过
                        continue

                    expr_value = int(float(expr_value))

                    gene_id = name_to_geneid.get(gene_name.lower(), None)
                    if gene_id is None:
                        continue

                    desc_index = id_to_index.get(gene_id, None)
                    if desc_index is None:
                        continue

                    gene_embedding = gene_desc_embeddings[desc_index]
                    weighted_embedding = gene_embedding * expr_value
                    cell_embedding += weighted_embedding

                return np.array(cell_embedding)

            # 并行处理所有细胞
            strings = cell_strings["CellString"].tolist()
            # tqdm + joblib 简单结合：先用普通列表再 tqdm 包外层
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_cell)(s) for s in tqdm(strings, desc="Generating weighted cell embeddings")
            )
            
            print("Converting results to numpy array...")
            cell_embeddings = np.array(results, dtype=object)  # 保持每个 cell 是 (<=top_k, dim)

            np.savez_compressed(
                weighted_cell_embeddings_save_path, embeddings=results
            )

        return np.array(cell_embeddings)