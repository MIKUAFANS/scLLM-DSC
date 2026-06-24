import os
import h5py
import numpy as np
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv
import scanpy as sc

from openai import OpenAI
from .utils import embedding_api

load_dotenv()

class GeneDescGenerator:
    def __init__(self, reference_path: str, dataset_path: str):
        self.reference_path = reference_path
        self.dataset_path = dataset_path
        self.openai_client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))

    def load_dataset(self, filename: str):
        if filename.endswith(".h5"):
            with h5py.File(os.path.join(self.dataset_path, filename), "r") as f:
                try:
                    if "gene_names" in f:
                        gene_names = f["gene_names"][...]
                    elif "var_names" in f:
                        gene_names = f["var_names"][...]
                    else:
                        gene_names = f["gene_name"][...]
                except KeyError:
                    print(f"Keys available in the dataset: {list(f.keys())}")
                    raise KeyError("The dataset does not contain 'gene_name' key.")
        elif filename.endswith(".h5ad"):
            adata = sc.read_h5ad(os.path.join(self.dataset_path, filename))
            gene_names = adata.var["feature_name"].values
        return gene_names

    def preprocess_reference(self, reference_file: str):
        reference = pd.read_csv(os.path.join(self.reference_path, reference_file))
        official_symbols = reference["Official_Symbol"].str.lower()
        also_known_as = reference["Also_known_as"].fillna('').str.lower().str.split(',\s*')
        gene_ids = reference[['GeneID', 'Official_Symbol']].copy()

        all_known_genes = set(official_symbols)
        for aliases in also_known_as:
            all_known_genes.update(aliases)
            
        gene_name_to_id = {official_symbols[i]: str(gene_ids['GeneID'].iloc[i]) for i in range(len(official_symbols))}
        for i, aliases in enumerate(also_known_as):
            for alias in aliases:
                gene_name_to_id[alias] = str(gene_ids['GeneID'].iloc[i])
        
        return all_known_genes, gene_name_to_id, reference

    def generate_gene_desc(self, reference: pd.DataFrame, dataset: pd.DataFrame):
        gene_desc = pd.DataFrame(columns=['GeneID', 'Description'])
        vaild_dataset = dataset.loc[dataset['ExistsInNCBI'] == True]
        vaild_dataset['GeneID'] = vaild_dataset['GeneID'].astype(int)
        for _, row in tqdm(vaild_dataset.iterrows(), total=vaild_dataset.shape[0], desc="Generating gene descriptions"):
            gene_id = int(row['GeneID'])
            desc = reference.loc[reference['GeneID'] == gene_id]
            desc_gene = f"Gene ID: {gene_id}. Gene Name: {desc['Official_Symbol'].values[0]}. Gene Full Name: {desc['Official_Full_Name'].values[0]}. Gene Type: {desc['Gene_type'].values[0]}. Organism: {desc['Organism_Scientific'].values[0]}. Organism: {desc['Organism_Scientific'].values[0]}. Also Known As: {desc['Also_known_as'].values[0]}. Lineage: {desc['Lineage'].values[0]}. Expression: {desc['Expression_Summary'].values[0]}. Primary Source: {desc['Primary_source'].values[0]}. Map Location: {desc['Map_Location'].values[0]}."
            
            gene_desc.loc[len(gene_desc)] = [row['GeneID'], desc_gene]
            
        return gene_desc

    def __call__(self, filename: str, save_path: str, reference_file: str, model: str="text-embedding-3-small") -> np.ndarray:
        all_known_genes, gene_name_to_id, reference = self.preprocess_reference(reference_file)
        dataset_save_path = os.path.join(save_path, "gene_descriptions_with_embeddings.csv")

        if os.path.exists(dataset_save_path):
            print(f"Gene descriptions with embeddings already exist at {dataset_save_path}. Loading existing file.")
            dataset = pd.read_csv(dataset_save_path)
        else:
            gene_names = self.load_dataset(filename)
            
            dataset = pd.DataFrame(gene_names.astype(str), columns=["gene_name"])
            gene_names = dataset["gene_name"].str.lower()
            print("Total genes:", len(gene_names))
            gene_names_not_in_reference = dataset[~gene_names.isin(all_known_genes)]["gene_name"].tolist()
            print("Genes not in reference:", len(gene_names_not_in_reference))
            
            with open(os.path.join(save_path, f"genes_not_in_reference.txt"), "w") as f:
                for gene in gene_names_not_in_reference:
                    f.write(gene + "\n")
                    
            dataset['ExistsInNCBI'] = gene_names.isin(all_known_genes)
            dataset['GeneID'] = gene_names.map(gene_name_to_id).fillna('NA')
            
            dataset.to_csv(dataset_save_path, index=False)

        gene_desc_dataset_path = os.path.join(save_path, "gene_desc.csv")
        if os.path.exists(gene_desc_dataset_path):
            print(f"{gene_desc_dataset_path} already exists. Skipping gene description generation.")
            gene_desc_dataset = pd.read_csv(gene_desc_dataset_path)
        else:
            gene_desc_dataset = self.generate_gene_desc(reference, dataset)
            gene_desc_dataset.to_csv(gene_desc_dataset_path, index=False)

        desc_embedding_save_path = os.path.join(save_path, f"gene_desc_embeddings.npz")
        if os.path.exists(desc_embedding_save_path):
            print(f"{desc_embedding_save_path} already exists. Skipping embedding generation.")
            embeddings = np.load(desc_embedding_save_path)['embeddings']
        else:
            gene_descriptions = gene_desc_dataset['Description'].tolist()
            embeddings = embedding_api(self.openai_client, gene_descriptions, model=model)

            np.savez_compressed(desc_embedding_save_path, embeddings=embeddings)
        
        return embeddings