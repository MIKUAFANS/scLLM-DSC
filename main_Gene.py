import argparse
import os

from model import GeneDescGenerator, GeneSelector, GeneEmbeddingWeighter

# Default paths relative to project root
DEFAULT_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "reference_data")
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets")
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--reference_path", type=str, default=DEFAULT_REFERENCE_PATH, help="Path to reference data")
    argparser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="Path to datasets")
    argparser.add_argument("--reference_file", type=str, default="human.csv", help="Reference gene data file")
    argparser.add_argument("--dataset", type=str, default="Sonya_HumanLiver_counts_top5000.h5", help="Dataset name")
    argparser.add_argument("--save_path", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save processed data")
    args = argparser.parse_args()

    filename = os.path.join(args.dataset_path, args.dataset)
    save_path = os.path.join(args.save_path, f"{os.path.basename(args.dataset).split('.')[0]}/")
    os.makedirs(save_path, exist_ok=True)

    gene_desc_generator = GeneDescGenerator(reference_path=args.reference_path, dataset_path=args.dataset_path)
    gene_desc_embeddings = gene_desc_generator(filename, save_path, args.reference_file)

    print("Gene description embeddings generated and saved.")
    print(gene_desc_embeddings.shape)

    gene_expression_generator = GeneSelector(filename=filename, save_path=save_path, n_jobs=64)
    cell_embeddings, _ = gene_expression_generator(save_path, top_k=2048)
    
    print("Cell embeddings generated and saved.")
    print(cell_embeddings.shape)
    
    weighted_cell_embeddings = GeneEmbeddingWeighter(save_path, n_jobs=64)()
    
    print("Weighted cell embeddings generated and saved.")
    print(weighted_cell_embeddings.shape)