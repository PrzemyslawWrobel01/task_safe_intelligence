import time
import torch
from torch.utils.data import Dataset
import pandas as pd
#Additional Imports
from multiprocess import Process, Queue #using multiprocess library, instead of multiprocessing, due to pickle limitations

class SingleProcessDataset(Dataset):
    def __init__(self, csv_file):
        start_time = time.time()
        print("Loading data using single process...")
        
        self.data = pd.read_csv(csv_file)
        self.features = torch.FloatTensor(self.data[['x1', 'x2', 'x3']].values)
        self.labels = torch.LongTensor(self.data['label'].values)
        
        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultiProcessDataset(SingleProcessDataset):

    def load_data_chunk(self,data,start_index,end_index,queue): #Target Method for multiprocessing, loading each chunk
            chunk_df = data.iloc[start_index:end_index]
            chunk_features = torch.FloatTensor(chunk_df[['x1','x2','x3']].values)
            chunk_labels = torch.LongTensor(chunk_df['label'].values)
            queue.put((chunk_features, chunk_labels))


    def __init__(self, csv_file):
        start_time = time.time()
        print("Loading data using multi process...")

        self.data = pd.read_csv(csv_file)
        self.processes = []
        self.num_of_processes=4
        self.queue = Queue() #Using queue to safely store data between processes
        self.features = []
        self.labels = []


        
        chunk_size = int(self.data.shape[0] / self.num_of_processes) #Calculating size of chunk
                    


        
        for i in range(self.num_of_processes):

            #Calculate start and end indices
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size
            #Create and start process
            p = Process(target=self.load_data_chunk, args=(self.data,start_index,end_index,self.queue))
            self.processes.append(p)
            p.start()
        
        #Collect results from each process
        for _ in range(self.num_of_processes):
            features, labels = self.queue.get()
            self.features.append(features)
            self.labels.append(labels)

        #Concatenate the features and labels from each process
        self.features = torch.cat(self.features)
        self.labels = torch.cat(self.labels)
        
        #Ensure all processes finished
        for p in self.processes:
            p.join()

        
        ########### END YOUR CODE  ############
        
        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")