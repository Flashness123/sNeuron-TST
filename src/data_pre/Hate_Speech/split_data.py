import argparse
import os
from tqdm import tqdm
from datasets import load_dataset

def main(args):
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    # Define output file paths
    train_original_file = open(os.path.join(args.out_path, 'train.original.txt'), 'w', encoding='utf-8')
    train_converted_file = open(os.path.join(args.out_path, 'train.converted.txt'), 'w', encoding='utf-8')
    test_original_file = open(os.path.join(args.out_path, 'test.original.txt'), 'w', encoding='utf-8')
    test_converted_file = open(os.path.join(args.out_path, 'test.converted.txt'), 'w', encoding='utf-8')

    # Load your private Hugging Face dataset
    total_dataset = load_dataset(args.data_hf, split="train", token="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")

    # Split into train (95%) and test (5%)
    split_dataset = total_dataset.train_test_split(test_size=0.05)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    # Process train_dataset
    for data in tqdm(train_dataset, desc="Processing train data"):
        train_original_file.write(data['Original Text'].strip() + '\n')
        train_converted_file.write(data['Converted Text'].strip() + '\n')
    
    # Process test_dataset
    for data in tqdm(test_dataset, desc="Processing test data"):
        test_original_file.write(data['Original Text'].strip() + '\n')
        test_converted_file.write(data['Converted Text'].strip() + '\n')

    # Close all files
    train_original_file.close()
    train_converted_file.close()
    test_original_file.close()
    test_converted_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_hf", type=str, default="Resi/hate_speech_v1", help="Hugging Face dataset name")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for split files")
    args = parser.parse_args()
    main(args)