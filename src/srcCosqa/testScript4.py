import os
from datetime import datetime
import numpy as np
import torch
from ProcessedDataset import ProcessedDataset
from config import args as arguments
from config import args_format as args_file_name
from model.GraphMatchModel import GraphMatchNetwork
from utils import write_log_file, arguments_to_tables, chunk

os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.gpu_index)


class Trainer(object):
    # Initialize Trainer object
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.data_dir

        if self.args.only_test:
            self.sig = os.path.join(args.log_dir, "OnlyTest_" + datetime.now().strftime("%Y-%m-%d@%H:%M:%S"))
        else:
            self.sig = os.path.join(args.log_dir, datetime.now().strftime("%Y-%m-%d@%H:%M:%S"))
        os.mkdir(self.sig)

        # Set paths for output files
        self.log_path = os.path.join(self.sig, 'log_{}.txt'.format(args_file_name))
        self.best_model_path = os.path.join(self.sig, 'best_model.pt')

        # Convert arguments to tables and write to log file
        table_draw = arguments_to_tables(args=arguments)
        write_log_file(self.log_path, str(table_draw))

        # Set batch sizes and maximum number of iterations
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.max_iteration = args.max_iter
        self.margin = args.margin

        # Set device to GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        write_log_file(self.log_path, "\n****CPU or GPU: " + str(self.device))

        # Set maximum number of edge types
        max_number_edge_types = 3

        # Initialize GraphMatchNetwork model
        if self.args.conv.lower() in ['rgcn', 'cg', 'nnconv']:
            self.model = GraphMatchNetwork(node_init_dims=300, arguments=args, device=self.device,
                                           max_number_of_edges=max_number_edge_types).to(self.device)
        else:
            raise NotImplementedError

        # Write model to log file
        write_log_file(self.log_path, str(self.model))

        # Set optimizer as Adam with given learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        # Initialize ProcessedDataset objects for code and text
        write_log_file(self.log_path, "Init Reading Code Graphs ... ")
        self.code_data = ProcessedDataset(name='code', root=self.dataset_dir, log_path=self.log_path,
                                          skip_check=self.args.skip_file_check,
                                          train_sample_size=self.args.train_sample_size,
                                          random_split=self.args.random_split)
        write_log_file(self.log_path, "Init Reading Text Graphs ... ")
        self.text_data = ProcessedDataset(name='text', root=self.dataset_dir, log_path=self.log_path,
                                          skip_check=self.args.skip_file_check,
                                          train_sample_size=self.args.train_sample_size,
                                          random_split=self.args.random_split)

        # for plotting and record (init empty list)
        self.train_iter, self.train_smooth_loss, self.valid_iter, self.valid_loss, self.test_iter, self.test_mrr, self.test_s1, self.test_s5, self.test_s10 = (
            [] for _ in range(9))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    # Henry: Rank candidate graphs
    def retrieve_rank(self, query_id, candidate_id_list, query_data, cand_data):
        # Initialize variables
        st = 0  # Henry: start
        rank = dict()
        one_query_scores = []

        # Iterate over candidate_id_list in batches of valid_batch_size
        while st < len(candidate_id_list):
            ed = st + self.valid_batch_size if st + self.valid_batch_size < len(candidate_id_list) else len(
                candidate_id_list)
            code_graph_list, text_graph_list = [], []
            for i in range(st, ed):
                code_graph_list.append(query_id)
                text_graph_list.append(candidate_id_list[i])
            code_batch = query_data.get_batch_graph(code_graph_list)
            text_batch = cand_data.get_batch_graph(text_graph_list)

            # Set the model to evaluation mode and perform a forward pass
            self.model.eval()
            with torch.no_grad():
                score = self.model(code_batch, text_batch)

            # Add the scores to the rank dictionary
            for candidate_id in text_graph_list:
                rank[candidate_id] = score[text_graph_list.index(candidate_id)]

            # Append the scores for the current query to one_query_scores list
            one_query_scores.extend(score.tolist())
            st = ed

        # Sort the rank dictionary in descending order based on the scores
        rank = [a[0] for a in sorted(list(rank.items()), key=lambda x: x[1], reverse=True)]

        # Ensure that the length of one_query_scores matches candidate_id_list
        assert len(one_query_scores) == len(candidate_id_list), "must be equal, ERROR"

        # Return the sorted rank list and the numpy array of scores for the current query
        return rank, np.array(one_query_scores)

    @staticmethod
    def calculate_square_mrr(similarity):
        # Ensure that the similarity matrix is square
        assert similarity.shape[0] == similarity.shape[1]

        # Extract the correct scores along the diagonal of the similarity matrix
        correct_scores = np.diagonal(similarity)

        # Generate a boolean matrix where elements are True if the similarity score is greater than or equal to the correct score
        # Henry: correct_scores has shape [n,], so we need to add a new axis to make its shape [n,1],
        #        then we can compare with similarity (shape [n,n])
        compared_scores = similarity >= correct_scores[..., np.newaxis]

        # Calculate the reciprocal rank for each row in the matrix
        # Henry: this formulae looks different from the one in the paper (sec. 4.3.2 eq.11)
        #        although it logically makes sense.
        rrs = 1.0 / compared_scores.astype(float).sum(-1)

        # Return the mean reciprocal rank
        return rrs

    def test(self, iter_no, test_chunk_size):
        # write log for starting testing
        write_log_file(self.log_path, "Start to testing ...")

        # get test query ids
        test_query_ids = self.text_data.split_ids['test']

        # initialize a dictionary to store success@1, success@5, and success@10
        success = {1: 0, 5: 0, 10: 0}

        # initialize an empty list to store all the scores for each test query
        total_test_scores = []

        # record the start time of testing
        test_start = datetime.now()

        # split test query ids into chunks of 100 and iterate over each chunk
        # Henry: chunk is imported from util file
        for test_chunk in chunk(test_query_ids, test_chunk_size):
            # initialize an empty list to store scores for each query in the current chunk
            one_chunk_scores = []
            for i, query_id in enumerate(test_chunk):
                print(i,query_id)
                # retrieve the rank and scores for the current query and store the scores
                if i != 135:
                    continue
                rank_ids, one_row_scores = self.retrieve_rank(query_id, test_chunk, self.text_data, self.code_data)

                if i == 135:
                    print(rank_ids)
                    exit()
                one_chunk_scores.append(one_row_scores)

                # update success dictionary if the current query is in top-k results
                for k in success.keys():
                    if query_id in rank_ids[:k]:
                        success[k] += 1

            # append the scores for the current chunk to the list of all scores
            total_test_scores.append(one_chunk_scores)

        # write log for finishing testing
        write_log_file(self.log_path,
                       "\n&Testing Iteration {}: for {} queries finished. Time elapsed = {}.".format(iter_no,
                                                                                                     len(test_query_ids),
                                                                                                     datetime.now() - test_start))

        # Henry: retrieve_rank returns a row, comparing each query in a chunk with all queries in the same chunk.
        #       chunk score stacks the rows to be a square matrix.
        all_mrr = []  # list to store all the MRR scores obtained from testing
        for i in range(len(total_test_scores)):  # iterate over each chunk of test scores
            one_chunk_square_score = total_test_scores[i]  # obtain one chunk of test scores
            one_chunk_square_score = np.vstack(
                one_chunk_square_score)  # stack the test scores vertically to create a square matrix
            assert one_chunk_square_score.shape[0] == one_chunk_square_score.shape[
                1], "Every Chunk must be square"  # check if the matrix is square
            mrr_array = self.calculate_square_mrr(one_chunk_square_score)  # calculate MRR for the given chunk of scores
            all_mrr.extend(mrr_array)  # add the MRR scores to the list of all MRR scores
        mrr = np.array(all_mrr).mean()  # calculate the average of all MRR scores
        self.test_iter.append(iter_no)  # append the current testing iteration number to the list of testing iterations
        self.test_mrr.append(mrr)  # append the calculated MRR to the list of MRR scores obtained during testing
        write_log_file(self.log_path, "&Testing Iteration {}: MRR = &{}&".format(iter_no,
                                                                                 mrr))  # write the testing iteration number and MRR to the log file


if __name__ == '__main__':
    all_time_1 = datetime.now()
    trainer = Trainer(arguments)
    if arguments.only_test:
        trainer.load_model(arguments.model_path)
    else:
        trainer.fit()
        trainer.load_model(trainer.best_model_path)

    all_time_1 = datetime.now()
    trainer.test(iter_no=trainer.max_iteration + 1, test_chunk_size=500)
