import argparse
import torch
import os


def main(args):
    style_list = [
        #'GYAFC', 'ParaDetox', 'Politics', 'Politness', 'Shakespeare', 'Yelp', 
        'Hate_Speech']
    style_dict = {
        # 'GYAFC': ['formal', 'informal'],
        # 'ParaDetox': ['toxic', 'neutral'],
        # 'Politics': ['democratic', 'republican'],
        # 'Politness': ['polite', 'impolite'],
        # 'Shakespeare': ['shakespeare', 'modern'],
        # 'Yelp': ['positive', 'negative'],
        'Hate_Speech': ['original', 'converted']

    }
    
    for style in style_list:
        print(f'process {style}')
        neurons = torch.load(args.data_path + f'/{style}.llama-7b')
        positive_neurons = neurons[0]
        negative_neurons = neurons[1]
                
        positive_neurons_list = [i.tolist() for i in positive_neurons]
        negative_neurons_list = [i.tolist() for i in negative_neurons]
        
        
        final_inter_tensor = []
        pos_non_inter_tensor = []
        neg_non_inter_tensor = []
        
        for _, (pos, neg) in enumerate(zip(positive_neurons_list, negative_neurons_list)):
            tmp_inter = list(set(pos).intersection(set(neg)))
            # intersection_list.append(tmp_inter)
            final_inter_tensor.append(torch.tensor(tmp_inter).long())
            
            tmp_pos_non_inter = list(set(pos).difference(set(neg)))
            # positive_non_inter.append(tmp_pos_non_inter)
            pos_non_inter_tensor.append(torch.tensor(tmp_pos_non_inter).long())
            
            tmp_neg_non_inter = list(set(neg).difference(set(pos)))
            # negative_non_inter.append(tmp_neg_non_inter)
            neg_non_inter_tensor.append(torch.tensor(tmp_neg_non_inter).long())
            
        
        ## 写文件
        inter_dir = os.path.join(args.data_path, 'inter_neurons')
        non_inter_dir = os.path.join(args.data_path, 'non_inter_neurons')
        os.makedirs(inter_dir, exist_ok=True)
        os.makedirs(non_inter_dir, exist_ok=True)
        
        torch.save(final_inter_tensor, args.data_path + f"/inter_neurons/{style}.inter_neurons.llama-7b")
        torch.save(pos_non_inter_tensor, args.data_path + f"/non_inter_neurons/{style}.{style_dict[style][0]}.llama-7b")
        torch.save(neg_non_inter_tensor, args.data_path + f"/non_inter_neurons/{style}.{style_dict[style][1]}.llama-7b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, default="data/Hate_Speech/neurons")
    args = parser.parse_args()
    main(args)
