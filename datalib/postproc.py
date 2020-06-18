"""
Author: Alex Katrompas

Post Processing Library
For time series binary classification data.
i.e. temporal classifaction
"""

import numpy as np

DEFAULT_THRESHHOLD = 0.7

class Postproc:
    
    
    ###############################
    # Constructor
    ###############################
    def __init__(self, filename, delim=","):
        """
        Constructor
        
        Parameters: filename to load
                    optional delimeter: comman, space, tab
        Processing: Call load_data
        Return: none
        """
        self.load_data(filename, delim=",")


    ###############################
    # Public
    ###############################
    def load_data(self, filename, delim=","):
        """
        Load and verify a data file with actual and predicted for binary classification.
        Can be called after creation to load a new file.
        
        Parameters: filename to load
                    optional delimeter: comman, space, tab
        Processing: Initialze all attributes, load the data, and verify it.
        Return: none
        """
        self.filename = filename
        self.data = []
        self.nsequences = 0
        self.largest_sequence = 0
        self.smallest_sequence = 0
        self.mse = []
        if delim != " " and delim != "," and delim != "\t":
            self.delim = ","
        else:
            self.delim = delim
        self.lines = self.__verify_file()
        if self.lines:
            self.__load_actual_predicted()
            self.loaded = True
        else:
            self.loaded = False


    ###############################
    # Display Fucntions
    ###############################
    def display_data(self):
        """
        Display all data
        
        Parameters: none
                    
        Processing: none (display function)
        Return: none
        """
        s_count = 0
        for sequence in self.data:
            print("Sequence:", s_count, "(", len(sequence), "pairs )")
            print(sequence)
            #for pair in sequence:
            #    print("  ", pair)
            s_count += 1


    def display_info(self):
        """
        Display all stats
        
        Parameters: none
                    
        Processing: none (display function)
        Return: none
        """
        print("Total Sequences:", self.nsequences)
        print("Largest Sequence:", self.largest_sequence)
        print("Smallest Sequence:", self.smallest_sequence)
        print("MSE per Sequence:")
        self.dispaly_mse()
        print()


    def dispaly_mse(self):
        """
        Display all MSE
        
        Parameters: none
                    
        Processing: none (display function)
        Return: none
        """
        ns = 0
        for mse in self.mse:
            ns += 1
            print("  ", ns, ":", mse)

    
    ###############################
    # Private
    ###############################
    def __verify_file(self):
        """
        Verify a csv has exactly two numeric values per line, and two classes total.

        Parameters: none
        Processing: will check all lines and compare total lines to good lines
                    will classes are first, and there are only two
                    i.e. it's checking actual verses predicted
        Return: T/F
        """
        fin = open(self.filename, "r")
        lines = goodlines = 0
        classes = {}

        for line in fin:
            line = line[:-1]
            if(len(line)):
                lines += 1
                line = line.split(self.delim)
                classes[line[0]] = 1
                try:
                    float(line[0])
                    float(line[1])
                    badvals = False
                except:
                    badvals = True
                if(len(line) == 2 and not badvals):
                    goodlines += 1

        fin.close()
                
        if lines == goodlines and len(classes) == 2:
            return lines
        else:
            return 0


    def __load_actual_predicted(self):
        """
        Loads the data into n sequences based on classes

        Parameters: none
        Processing: The data is loaded into sequenses based on binary
                    classification. Each sequence is a class, in order.
        Return: nothing.
        """
        fin = open(self.filename, "r")

        # get first line
        line = fin.readline()[:-1]
        line = line.split(self.delim)
        line[0] = float(line[0])
        line[1] = float(line[1])
        
        # set first classifcation
        classification = line[0]
        # start first sequence
        sequence = [line]
        
        for line in fin:
            line = line[:-1]
            if(len(line)):
                line = line.split(self.delim)
                line[0] = float(line[0])
                line[1] = float(line[1])
                if line[0] == classification:
                    sequence.append(line)
                else:
                    # save sequence
                    self.data.append(sequence)
                    # set next classifcation
                    classification = line[0]
                    # start next seqence
                    sequence = [line]

        # save last sequence
        self.data.append(sequence)
        
        self.__set_info()
        self.nsequences = len(self.data)
        fin.close()


    def __set_info(self):
        """
        calculates and sets all stats for the loaded data
        
        Parameters: none
        Processing: calculates and sets all stats for the loaded data
        Return: none
        """
        self.nsequences = len(self.data)
        
        maxs = len(self.data[0])
        mins = len(self.data[0])
        for sequence in self.data:
            length = len(sequence)
            if length > maxs:
                maxs = length
            if length < mins:
                mins = length
        self.largest_sequence = maxs
        self.smallest_sequence = mins
        self.calc_mse()


    def __calc_mse(self):
        """
        Calculate MSE for all actual and predicted sequences
        
        Parameters: none
                    
        Processing: Calculate MSE for all actual and predicted sequences
        Return: none
        """
        for sequence in self.data:
            s = np.array(sequence)
            s = s.transpose()
            self.mse.append(np.mean((s[0] - s[1])**2))

    