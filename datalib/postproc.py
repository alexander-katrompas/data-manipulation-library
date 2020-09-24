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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

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
        self.DEFAULT_THRESHHOLD = 0.50
        self.DEFAULT_PERCENTAGE = 0.50
        self.TP = 1
        self.FP = 2
        self.TN = 3
        self.FN = 4
        
        self.load_data(filename, delim=",")


    ###############################
    # Public
    ###############################

    def get_avg_mse(self):
        """
        Returns average MSE
        
        Parameters: none
        Processing: Calculates and returns average MSE
        Return: none
        """
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
        self.total_avp = 0
        self.nsequences = 0
        self.largest_sequence = 0
        self.smallest_sequence = 0
        self.mse = []
        self.marked = []
        self.tp = self.fp = self.tn = self.fn = -1
        self.tp1 = self.fp1 = self.tn1 = self.fn1 = -1
        self.roc_auc = 0 # need to call graph_roc to set
        self.pr_auc = 0 # need to call graph_pr to set
        
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
        """
        Returns or prints true positives.
        
        Parameters: Optional print flag to flip between print and return
        Processing: Extracts true positives, puts them in a dicionary with the
                    sequence number as the key (not sequential, it's a key)
        Return: Optionally the dictionary created.
        """
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
        """
        Returns or prints false positives.
        
        Parameters: Optional print flag to flip between print and return
        Processing: Extracts false positives, puts them in a dicionary with the
                    sequence number as the key (not sequential, it's a key)
        Return: Optionally the dictionary created.
        """
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
        """
        Returns or prints true negatives.
        
        Parameters: Optional print flag to flip between print and return
        Processing: Extracts true negatives, puts them in a dicionary with the
                    sequence number as the key (not sequential, it's a key)
        Return: Optionally the dictionary created.
        """
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
        """
        Returns or prints false negatives.
        
        Parameters: Optional print flag to flip between print and return
        Processing: Extracts false negatives, puts them in a dicionary with the
                    sequence number as the key (not sequential, it's a key)
        Return: Optionally the dictionary created.
        """
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
        """
        Returns the ROC area under the cruve
        
        Parameters: none
        Processing: none
        Return: none
        """
        return self.roc_auc


    def get_pr_auc(self):
        """
        Returns the RR area under the cruve
        
        Parameters: none
        Processing: none
        Return: none
        """
        return self.pr_auc


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
            print("Total AvP:", self.total_avp)
            print("Total Sequences:", self.nsequences)
            print("Largest Sequence:", self.largest_sequence)
            print("Smallest Sequence:", self.smallest_sequence)
            if mse:
                print("MSE per Sequence:")
                self.dispaly_mse()
            else:
                print("Average MSE: {:.3f}".format(self.get_avg_mse()))
            print()
            self.display_confusion_matrix()
            self.display_seq_confusion_matrix()
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
        """
        Display the confusion matrix
        
        Parameters: none
                    
        Processing: none (display function)
        Return: none
        """
        print("=========AvP CONFUSION MATRIX=========")
        print(" -------------------------------------")
        print(" |{:>12}{:>12}{:>12}".format("C.MATRIX |", "Pos Pred |", "Neg Pred |"))
        print(" -------------------------------------")
        print(" |{:>12}{:>12}{:>12}".format("Pos Class |", str(self.tp1) + "    |", str(self.fn1) + "    |"))
        print(" -------------------------------------")
        print(" |{:>12}{:>12}{:>12}".format("Neg Class |", str(self.fp1) + "    |", str(self.tn1) + "    |"))
        print(" -------------------------------------")
        total = self.tp1 + self.fp1 + self.fn1 + self.tn1
        if self.total_avp == total:
            match = "yes"
        else:
            match = "no"
        print("    |tp|fn|    Total AvP: {}".format(self.total_avp))
        print("    |-----|    Total Cases: {}".format(total))
        print("    |fp|tn|    Match: {}".format(match))
        print()

        accuracy = (self.tp1 + self.tn1) / total
        print("Accuracy:", accuracy)
        
        precision = self.tp1 / (self.tp1 + self.fp1)
        print("Precision:", precision)
        
        recall = self.tp1 / (self.tp1 + self.fn1)
        print("Recall:", recall)

        f1 = 2 * (recall * precision) / (recall + precision)
        print("F1:", f1)
        print()

    
    def display_seq_confusion_matrix(self):
        """
        Display the confusion matrix
        
        Parameters: none
                    
        Processing: none (display function)
        Return: none
        """
        print("======SEQUENCE CONFUSION MATRIX=======")
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

        accuracy = (self.tp + self.tn) / total_cases
        print("Accuracy:", accuracy)
        
        div = (self.tp + self.fp)
        if div != 0:
            precision = self.tp / (self.tp + self.fp)
        else:
            precision = 0
        print("Precision:", precision)
        
        div = (self.tp + self.fn)
        if div != 0:
            recall = self.tp / (self.tp + self.fn)
        else:
            recall = 0
        print("Recall:", recall)

        div = (recall + precision)
        if div != 0:
            f1 = 2 * (recall * precision) / (recall + precision)
        else:
            f1 = 0
        print("F1:", f1)
        print()
    
    
    def display_marked(self):
        """
        Display all sequences amd how they are marked TP/FP/TN/FN
        
        Parameters: none
                    
        Processing: none (display function)
        Return: none
        """
        for i in range(self.nsequences):
            print("{}: {}".format(i, self.marked[i]))
    
    def display_sequence(self, i):
        """
        Display a specific sequence
        
        Parameters: the index of the sequence
                    
        Processing: none (display function)
        Return: none
        """
        if i < self.nsequences:
            print(self.sequenced_data[i])
        else:
            print("Sequence index out of range 0-{}".format(self.nsequences))
    
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

    def graph_pr(self):
        """
        precision recall curve
        
        """
        y = self.data[:, 0]
        yhat = self.data[:, 1]
        
        # calculate the no skill line as the proportion of the positive class
        no_skill = len(y[y==1]) / len(y)
        # plot the no skill precision-recall curve
        pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        # calculate model precision-recall curve
        precision, recall, _ = precision_recall_curve(y, yhat)
        # plot the model precision-recall curve
        pyplot.plot(recall, precision, marker='.', label='Model')
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()    
    
        self.pr_auc = auc(recall, precision)
        return self.pr_auc
    
    
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

        self.data = np.array(self.data)
        self.__set_info()
        
        fin.close()


    def __set_info(self):
        """
        calculates and sets all stats for the loaded data
        
        Parameters: none
        Processing: calculates and sets all stats for the loaded data
        Return: none
        """
        self.nsequences = len(self.sequenced_data)
        
        self.total_avp = len(self.data)
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
        self.__calc_matrix_seq()


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
        Calculate a confusion matrix based on 1 for 1 actual versus predicted
        
        Parameters: none
                    
        Processing: Calculate a confusion matrix based on 1 for 1
                    actual versus predicted
        Return: none
        """
        self.tp1 = self.tn1 = self.fp1 = self.fn1 = 0
    
        for pair in self.data:
            if pair[0] == 0 and pair[1] < self.DEFAULT_THRESHHOLD:
                self.tn1 += 1
            elif pair[0] == 0 and pair[1] > self.DEFAULT_THRESHHOLD:
                self.fn1 += 1
            elif pair[0] == 1 and pair[1] > self.DEFAULT_THRESHHOLD:
                self.tp1 += 1
            elif pair[0] == 1 and pair[1] < self.DEFAULT_THRESHHOLD:
                self.fp1 += 1
    
    def __calc_matrix_seq(self):
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
