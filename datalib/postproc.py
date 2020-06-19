"""
Author: Alex Katrompas

Post Processing Library
For time series binary classification data.
i.e. temporal classifaction
"""

from matplotlib import pyplot
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

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
        self.DEFAULT_THRESHHOLD = 0.51
        self.DEFAULT_PERCENTAGE = 0.51
        self.TP = 1
        self.FP = 2
        self.TN = 3
        self.FN = 4

        self.load_data(filename, delim=",")


    ###############################
    # Public
    ###############################
    def get_avg_mse(self):
        return float(sum(self.mse)) / float(self.nsequences)
    
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
        self.data = [] # will become a numpy array
        self.sequenced_data = []
        self.nsequences = 0
        self.largest_sequence = 0
        self.smallest_sequence = 0
        self.mse = []
        self.marked = []
        self.tp = self.fp = self.tn = self.fn = -1
        self.roc_auc = 0 # need to call graph_roc to set
        
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

    def get_tp(self, printit=False):
        marked = {}
        for i in range(self.nsequences):
            if self.marked[i] == self.TP:
                marked[i] = self.sequenced_data[i]
        
        if printit:
            for key in marked:
                print(key)
                for val in marked[key]:
                    print("  ", val)
                print()
        else:
            return marked

    def get_fp(self, printit=False):
        marked = {}
        for i in range(self.nsequences):
            if self.marked[i] == self.FP:
                marked[i] = self.sequenced_data[i]
        if printit:
            for key in marked:
                print(key)
                for val in marked[key]:
                    print("  ", val)
                print()
        else:
            return marked

    def get_tn(self, printit=False):
        marked = {}
        for i in range(self.nsequences):
            if self.marked[i] == self.TN:
                marked[i] = self.sequenced_data[i]
        if printit:
            for key in marked:
                print(key)
                for val in marked[key]:
                    print("  ", val)
                print()
        else:
            return marked

    def get_fn(self, printit=False):
        marked = {}
        for i in range(self.nsequences):
            if self.marked[i] == self.FN:
                marked[i] = self.sequenced_data[i]
        if printit:
            for key in marked:
                print(key)
                for val in marked[key]:
                    print("  ", val)
                print()
        else:
            return marked


    def get_roc_auc(self):
        return self.roc_auc

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
        if self.loaded:
            s_count = 0
            for sequence in self.sequenced_data:
                print("Sequence:", s_count, "(", len(sequence), "pairs )")
                print(sequence)
                #for pair in sequence:
                #    print("  ", pair)
                s_count += 1
        else:
            print("No data loaded.")


    def display_info(self, mse=True):
        """
        Display all stats
        
        Parameters: none
                    
        Processing: none (display function)
        Return: none
        """
        if self.loaded:
            print("Total Sequences:", self.nsequences)
            print("Largest Sequence:", self.largest_sequence)
            print("Smallest Sequence:", self.smallest_sequence)
            if mse:
                print("MSE per Sequence:")
                self.dispaly_mse()
            else:
                print("Average MSE: {:.3f}".format(self.get_avg_mse()))
            self.display_confusion_matrix()
            print()
        else:
            print("No data loaded.")


    def dispaly_mse(self):
        """
        Display all MSE
        
        Parameters: none
                    
        Processing: none (display function)
        Return: none
        """
        if self.loaded:
            ns = 0
            for mse in self.mse:
                ns += 1
                print("{}: {:.3f}".format(ns, mse))
        else:
            print("No data loaded.")

    def display_confusion_matrix(self):
        print()
        print(" -------------------------------------")
        print(" |{:>12}{:>12}{:>12}".format("C.MATRIX |", "Pos Pred |", "Neg Pred |"))
        print(" -------------------------------------")
        print(" |{:>12}{:>12}{:>12}".format("Pos Class |", str(self.tp) + "    |", str(self.fn) + "    |"))
        print(" -------------------------------------")
        print(" |{:>12}{:>12}{:>12}".format("Neg Class |", str(self.fp) + "    |", str(self.tn) + "    |"))
        print(" -------------------------------------")
        total_cases = self.tp + self.fp + self.fn + self.tn
        if self.nsequences == total_cases:
            match = "yes"
        else:
            match = "no"
        print("    |tp|fn|    Total Sequences: {}".format(self.nsequences))
        print("    |-----|    Total Cases: {}".format(total_cases))
        print("    |fp|tn|    Match: {}".format(match))
        print()
    
    
    def display_marked(self):
        for i in range(self.nsequences):
            print("{}: {}".format(i, self.marked[i]))
    
    def display_sequence(self, i):
        print(self.sequenced_data[i])
    
    ###############################
    # Graph Functions
    ###############################
    
    def graph_roc(self):
        """
        receiver operating characteristic curve
        
        """
        y = self.data[:, 0]
        yhat = self.data[:, 1]

        # plot no skill roc curve
        pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        # calculate roc curve for model
        fpr, tpr, _ = roc_curve(y, yhat)
        # plot model roc curve
        pyplot.plot(fpr, tpr, marker='.', label='Model')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

        self.roc_auc = roc_auc_score(y, yhat)
        return self.roc_auc
    
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
        
        # add data to flat list
        self.data.append(line)
        
        for line in fin:
            line = line[:-1]
            if(len(line)):
                line = line.split(self.delim)
                line[0] = float(line[0])
                line[1] = float(line[1])

                # add data to flat list
                self.data.append(line)
                
                if line[0] == classification:
                    sequence.append(line)
                else:
                    # save sequence
                    self.sequenced_data.append(sequence)
                    # set next classifcation
                    classification = line[0]
                    # start next seqence
                    sequence = [line]

        # save last sequence
        self.sequenced_data.append(sequence)
        
        self.nsequences = len(self.sequenced_data)
        self.marked = [0] * self.nsequences
        self.__set_info()
        
        fin.close()
        
        self.data = np.array(self.data)

    def __set_info(self):
        """
        calculates and sets all stats for the loaded data
        
        Parameters: none
        Processing: calculates and sets all stats for the loaded data
        Return: none
        """
        self.nsequences = len(self.sequenced_data)
        
        maxs = len(self.sequenced_data[0])
        mins = len(self.sequenced_data[0])
        for sequence in self.sequenced_data:
            length = len(sequence)
            if length > maxs:
                maxs = length
            if length < mins:
                mins = length
        self.largest_sequence = maxs
        self.smallest_sequence = mins
        self.__calc_mse()
        self.__calc_matrix()


    def __calc_mse(self):
        """
        Calculate MSE for all actual and predicted sequences
        
        Parameters: none
                    
        Processing: Calculate MSE for all actual and predicted sequences
        Return: none
        """
        for sequence in self.sequenced_data:
            s = np.array(sequence)
            s = s.transpose()
            self.mse.append(np.mean((s[0] - s[1]) ** 2))

    def __calc_matrix(self):
        """
        Calculate a confusion matrix based on sequences
        
        Parameters: none
                    
        Processing: Calculate a confusion matrix based on sequences by counting
                    all TP/FP/TN/FN and then applying a percent to determine
                    the "truth" of the whole sequence.
        Return: none
        """
        self.tp = self.tn = self.fp = self.fn = 0
        
        count = 0
        for sequence in self.sequenced_data:
            scount = len(sequence)
            target = sequence[0][0] # the first actual
            tmp_tp = tmp_fp = tmp_tn = tmp_fn = 0
            
            
            if target == 1:
                for i in range(scount):
                    if sequence[i][1] > self.DEFAULT_THRESHHOLD:
                        tmp_tp += 1
                    else:
                        tmp_fn += 1
                percent_tp = float(tmp_tp) / float(scount)
                #percent_fn = float(tmp_fn) / float(scount)
                if percent_tp >= self.DEFAULT_PERCENTAGE:
                    self.tp += 1
                    self.marked[count] = self.TP
                else:
                    self.fn += 1
                    self.marked[count] = self.FN
                
            else:
                for i in range(scount):
                    if sequence[i][1] < self.DEFAULT_THRESHHOLD:
                        tmp_tn += 1
                    else:
                        tmp_fp += 1
                percent_tn = float(tmp_tn) / float(scount)
                #percent_fp = float(tmp_fp) / float(scount)
                if percent_tn >= self.DEFAULT_PERCENTAGE:
                    self.tn += 1
                    self.marked[count] = self.TN
                else:
                    self.fp += 1
                    self.marked[count] = self.FP
            count += 1
