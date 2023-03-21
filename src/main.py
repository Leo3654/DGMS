#chatGPT comments

#This code is a Python script that defines a class Trainer used for training a graph matching model. 
#It imports necessary modules, initializes the parameters, and defines functions for training, validation, and testing.

#The Trainer class takes args as an argument, which contains the hyperparameters used to initialize the class. 
#It then initializes various parameters, such as the dataset directory, batch sizes, maximum iteration, margin, and device (CPU or GPU).

#It defines a model called GraphMatchNetwork, which is a graph matching model that takes the node dimensions, arguments, device, 
#and maximum number of edges as inputs.

#The fit function is used to train the model. It initializes the best validation loss and empty lists for various performance metrics.
#It then computes similarity between the positive code graph and text graph pairs, and between negative code graph and text graph pairs. 
#It uses the similarity to calculate the loss and optimizes the model parameters using the Adam optimizer. 
#It appends the loss to the list of losses and prints the mean smooth loss at every print_interval iterations. 
#It also performs validation and saves the model with the best validation loss. Finally, it performs testing on the trained model.

#The code also contains some utility functions to write log files, chunk data, and draw tables.


#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import required libraries
import os
from datetime import datetime
import numpy as np
import torch
from ProcessedDataset import ProcessedDataset
from config import args as arguments
from config import args_format as args_file_name
from model.GraphMatchModel import GraphMatchNetwork
from utils import write_log_file, arguments_to_tables, chunk

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.gpu_index)

# Define Trainer class
class Trainer(object):
    # Initialize Trainer object
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.data_dir
        
        # Create output directory based on current time
        # Henry: test or text? It should be test
        if self.args.only_test:
            self.sig = os.path.join(args.log_dir, "OnlyText_" + datetime.now().strftime("%Y-%m-%d@%H:%M:%S"))
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
            self.model = GraphMatchNetwork(node_init_dims=300, arguments=args, device=self.device, max_number_of_edges=max_number_edge_types).to(self.device)
        else:
            raise NotImplementedError
        
        # Write model to log file
        write_log_file(self.log_path, str(self.model))
        
        # Set optimizer as Adam with given learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        
        # Initialize ProcessedDataset objects for code and text     
        write_log_file(self.log_path, "Init Reading Code Graphs ... ")
        self.code_data = ProcessedDataset(name='code', root=self.dataset_dir, log_path=self.log_path)
        write_log_file(self.log_path, "Init Reading Text Graphs ... ")
        self.text_data = ProcessedDataset(name='text', root=self.dataset_dir, log_path=self.log_path)
        
        # for plotting and record (init empty list)
        self.train_iter, self.train_smooth_loss, self.valid_iter, self.valid_loss, self.test_iter, self.test_mrr, self.test_s1, self.test_s5, self.test_s10 = ([] for _ in range(9))
    
    def fit(self):
        best_val_loss = 1e10  # initialize the best validation loss to a very high value
        all_loss = []  # list to store all losses
        code_train_batch = self.code_data.triple_train_batch(self.train_batch_size)  # get a generator object for training batches
        time_1 = datetime.now()  # get the current time
        for iteration in range(self.max_iteration):  # loop through each iteration
            self.model.train()  # set the model to train mode
            # Compute similarity between positive and negative samples
            pos_code_graph_id_list, text_graph_id_list, neg_code_graph_id_list = next(code_train_batch)  # get a new batch of training data from the generator
            pos_code_batch = self.code_data.get_batch_graph(pos_code_graph_id_list)  # get the batch of positive code graphs
            text_batch = self.text_data.get_batch_graph(text_graph_id_list)  # get the batch of text graphs
            neg_code_batch = self.code_data.get_batch_graph(neg_code_graph_id_list)  # get the batch of negative code graphs
        
            pos_pred = self.model(pos_code_batch, text_batch).reshape(-1, 1)  # forward pass for positive samples
            neg_pred = self.model(neg_code_batch, text_batch).reshape(-1, 1)  # forward pass for negative samples
        
            #computes the triplet loss, which penalizes the model if the similarity between the positive code and text samples is less than 
            #the similarity between the negative code and text samples by at least a margin of self.margin
            #Henry: clamp() is used here to control the loss greater than 1e-6, so numerical instability or vanishing gradients.
            loss = (self.margin - pos_pred + neg_pred).clamp(min=1e-6).mean()  # calculate the contrastive loss
        
            self.optimizer.zero_grad()  # reset the gradients
            loss.backward()  # backpropagate the loss
            self.optimizer.step()  # update the weights of the model
            all_loss.append(loss)  # add the loss to the list of losses
        
            # Print training progress
            if iteration % self.args.print_interval == 0 and iteration > 0:  # if it is time to print progress
                self.train_iter.append(iteration)  # add the iteration number to the list of training iterations
                #Henry: cpu() is for cpu operation, mean() takes the mean of all_loss values of the print interval, detach() prevent further gradient computations.
                self.train_smooth_loss.append(torch.tensor(all_loss).mean().cpu().detach())  # add the smoothed loss to the list of training losses
                write_log_file(self.log_path, '@Train Iter {}: mean smooth loss = @{}@, time = {}.'.format(iteration, torch.tensor(all_loss).mean(), datetime.now() - time_1))  # write the training progress to the log file
                all_loss = []  # reset the list of losses
                time_1 = datetime.now()  # reset the time counter
        
            # Validation
            if (iteration % self.args.valid_interval == 0 and iteration >= self.args.val_start) or iteration == 0:  # if it is time to validate
                s_time = datetime.now()  # start the timer
                loss = self.validation()  # compute the validation loss
                # Record current iteration number, and validation loss
                self.valid_iter.append(iteration)
                self.valid_loss.append(loss.cpu().detach())
                # Get end time after validation
                end_time = datetime.now()
                # If the validation loss is smaller than the best validation loss seen so far
                if loss < best_val_loss:
                    # Update the best validation loss
                    best_val_loss = loss
                    # Save the model's state dictionary to the best model path
                    torch.save(self.model.state_dict(), self.best_model_path)
                    # Write the log message with the new best validation loss and time elapsed
                    write_log_file(self.log_path, '#Valid Iter {}: loss = #{}# (Decrease) < Best loss = {}. Save to best model..., time elapsed = {}.'.format(iteration, loss, best_val_loss, end_time - s_time))
                else:
                    # Write the log message with the current validation loss and time elapsed
                    write_log_file(self.log_path, '#Valid Iter {}: loss = #{}# (Increase). Best val loss = {}, time elapsed = {}.'.format(iteration, loss, best_val_loss, end_time - s_time))

            # only testing when iteration == 0 (whether code is rightly run)
            if iteration == 0:
                self.test(iter_no=iteration)

    def validation(self):
        """
        Perform a validation using code as base data.
        :return: mean validation loss over the whole validation set.
        """
        with torch.no_grad(): # context manager to avoid calculating gradients
            self.model.eval() # put the model in evaluation mode
            val_loss = [] # create an empty list to store validation loss
            # iterate over each batch of triplets in the validation dataset
            for pos_code_gid_list, text_gid_list, neg_code_gid_list in self.code_data.triple_valid_batch(self.valid_batch_size):
                # get the graph data for the positive, text, and negative codes in the triplet
                pos_code_batch = self.code_data.get_batch_graph(pos_code_gid_list)
                text_batch = self.text_data.get_batch_graph(text_gid_list)
                neg_code_batch = self.code_data.get_batch_graph(neg_code_gid_list)
            
                # calculate the similarity score between the positive and text, and negative and text codes
                pos_pred = self.model(pos_code_batch, text_batch).reshape(-1, 1)
                neg_pred = self.model(neg_code_batch, text_batch).reshape(-1, 1)
            
                # calculate the validation loss for this triplet
                loss = (self.margin - pos_pred + neg_pred).clamp(min=1e-6).mean()
                val_loss.append(loss) # add the validation loss to the list of losses
            # calculate the mean validation loss over the whole validation set
            loss = torch.tensor(val_loss).mean()
        return loss

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    # Henry: Rank candidate graphs
    def retrieve_rank(self, query_id, candidate_id_list, query_data, cand_data):
        # Initialize variables
        st = 0 # Henry: start
        rank = dict()
        one_query_scores = []
        
        # Iterate over candidate_id_list in batches of valid_batch_size
        while st < len(candidate_id_list):
            ed = st + self.valid_batch_size if st + self.valid_batch_size < len(candidate_id_list) else len(candidate_id_list)
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
        rrs = 1.0 / compared_scores.astype(np.float).sum(-1)
    
        # Return the mean reciprocal rank
        return rrs

    def test(self, iter_no):
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
        for test_chunk in chunk(test_query_ids, 100):
            # initialize an empty list to store scores for each query in the current chunk
            one_chunk_scores = []
            for i, query_id in enumerate(test_chunk):
                # retrieve the rank and scores for the current query and store the scores
                rank_ids, one_row_scores = self.retrieve_rank(query_id, test_chunk, self.text_data, self.code_data)
                one_chunk_scores.append(one_row_scores)

                # update success dictionary if the current query is in top-k results
                for k in success.keys():
                    if query_id in rank_ids[:k]:
                        success[k] += 1

            # append the scores for the current chunk to the list of all scores
            total_test_scores.append(one_chunk_scores)

        # write log for finishing testing
        write_log_file(self.log_path, "\n&Testing Iteration {}: for {} queries finished. Time elapsed = {}.".format(iter_no, len(test_query_ids), datetime.now() - test_start))
        
        # Henry: retrieve_rank returns a row, comparing each query in a chunk with all queries in the same chunk. 
        #       chunk score stacks the rows to be a square matrix.
        all_mrr = []  # list to store all the MRR scores obtained from testing
        for i in range(len(total_test_scores)):  # iterate over each chunk of test scores
            one_chunk_square_score = total_test_scores[i]  # obtain one chunk of test scores
            one_chunk_square_score = np.vstack(one_chunk_square_score)  # stack the test scores vertically to create a square matrix
            assert one_chunk_square_score.shape[0] == one_chunk_square_score.shape[1], "Every Chunk must be square"  # check if the matrix is square
            mrr_array = self.calculate_square_mrr(one_chunk_square_score)  # calculate MRR for the given chunk of scores
            all_mrr.extend(mrr_array)  # add the MRR scores to the list of all MRR scores
        mrr = np.array(all_mrr).mean()  # calculate the average of all MRR scores
        self.test_iter.append(iter_no)  # append the current testing iteration number to the list of testing iterations
        self.test_mrr.append(mrr)  # append the calculated MRR to the list of MRR scores obtained during testing
        write_log_file(self.log_path, "&Testing Iteration {}: MRR = &{}&".format(iter_no, mrr))  # write the testing iteration number and MRR to the log file

        for k, v in success.items():  # iterate over each key-value pair in the dictionary of success counts
            value = v * 1.0 / len(test_query_ids)  # calculate the success rate for the given value
            write_log_file(self.log_path, "&Testing Iteration {}: S@{}@ = &{}&".format(iter_no, k, value))  # write the testing iteration number, success rate key and value to the log file
            if k == 1:  # if the success rate key is 1, append the success rate to the list of S@1 scores obtained during testing
                self.test_s1.append(value)
            elif k == 5:  # if the success rate key is 5, append the success rate to the list of S@5
                self.test_s5.append(value)
            elif k == 10:
                self.test_s10.append(value)
            else:
                print('cannot find !')
        write_log_file(self.log_path, "S@1, S@5, S@10\n{}, {}, {}".format(self.test_s1[-1], self.test_s5[-1], self.test_s10[-1]))


if __name__ == '__main__':
    all_time_1 = datetime.now()
    trainer = Trainer(arguments)
    if arguments.only_test:
        trainer.load_model(arguments.model_path)
    else:
        trainer.fit()
        trainer.load_model(trainer.best_model_path)
    
    all_time_1 = datetime.now()
    write_log_file(trainer.log_path, "finished to load the model, next to start to test and time is = {}".format(all_time_1))
    trainer.test(iter_no=trainer.max_iteration + 1)
    write_log_file(trainer.log_path, "\nAll Finished using ({})\n".format(datetime.now() - all_time_1))

