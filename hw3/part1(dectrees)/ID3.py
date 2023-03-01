import numpy as np
import time
# In the decision tree, non-leaf nodes are going to be represented via TreeNode
import os


class TreeNode:
    def __init__(self, attribute: int):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}

# In the decision tree, leaf nodes are going to be represented via TreeLeafNode


class TreeLeafNode:
    def __init__(self,  label, numberOfInstances):
        #self.data = data
        self.label = label
        self.count = numberOfInstances


def log(base: np.array, operand: np.array):
    '''Because numpy does not have log with a custom base, we use an alias function'''
    return np.log(operand)/np.log(base)


class DecisionTree:
    def getCriterion(self, criterion: str):
        if criterion == "information gain":
            return self.calculate_information_gain__
        elif criterion == "gain ratio":
            return self.calculate_gain_ratio__
        if type(criterion) == str:
            raise ValueError(
                f"The parameter criterion {criterion} is invalid, should be either 'information gain' or 'gain ratio'"
            )
        raise ValueError(f"Type of criterion {type(criterion)} is not 'str'")

    def __init__(self, dataset, labels, attributes, classes=None, criterion="information gain"):
        if classes == None:
            self.classes = np.unique(labels)
        else:
            self.classes = classes
        self.base = len(self.classes)
        criterion = self.getCriterion(criterion)

        self.dataset = dataset
        self.labels = labels
        self.attributes = attributes
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

    def calculate_entropy__(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts/(counts.sum())
        return -((p*log(self.base, p)).sum())

    def calculate_average_entropy__(self, dataset, labels, attribute: int):
        average_entropy = 0
        valuesOfA = np.unique(dataset[:, attribute])
        # for each value v_i of A (attribute):
        for _value in valuesOfA:
            _filter = (dataset == _value).any(axis=1)
            # get y_i, all examples in y with attribute v_i for its length
            sLabels = labels[_filter]
            average_entropy += (
                (len(sLabels)/len(labels)) *
                self.calculate_entropy__(sLabels)
            )
        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute: int):
        return self.calculate_entropy__(labels)-self.calculate_average_entropy__(dataset, labels, attribute)

    def calculate_intrinsic_information__(self, dataset, labels, attribute: int):
        intrinsic_info = 0
        valuesOfA = np.unique(dataset[:, attribute])
        # for each value v_i of A (attribute):
        for _value in valuesOfA:
            _filter = (dataset == _value).any(axis=1)
            # get y_i, all examples in y with attribute v_i for its length
            sLabels = labels[_filter]
            operand = len(sLabels)/len(labels)
            intrinsic_info -= operand*log(self.base, operand)
        return intrinsic_info

    def calculate_gain_ratio__(self, dataset, labels, attribute: int):
        return self.calculate_information_gain__(dataset, labels, attribute)/self.calculate_intrinsic_information__(dataset, labels, attribute)

    def ID3__(self, dataset: np.array, labels: np.array, unusedAttributes: np.array):
        # if all examples in S belong to the same class C_j, return $S$ as a leaf node
        for _class in self.classes:
            if (labels == _class).all():
                return TreeLeafNode(_class, len(labels))
        # else select an attribute A according to some heuristic function h
        maxAttrVal = -np.inf
        maxAttrIndex = None
        for i in unusedAttributes:
            val = self.criterion(dataset, labels, i)
            if val > maxAttrVal:
                maxAttrVal = val
                maxAttrIndex = i
        unusedAttributesNew = unusedAttributes[unusedAttributes != maxAttrIndex]
        # create a decision tree DT with only one node with the attribute
        DT: TreeNode = TreeNode(maxAttrIndex)
        # for each value v_i of A:
        valuesOfA = np.unique(dataset[:, maxAttrIndex])
        for _value in valuesOfA:
            # get S_i, all examples in S with attribute v_i
            _filter = (dataset == _value).any(axis=1)
            sDataset = dataset[_filter]
            # similarly y_i, all examples in y with attribute v_i
            sLabels = labels[_filter]
            # obtain DT_i = ID3(S_i, y_i)
            DTchild = self.ID3__(
                sDataset, sLabels, unusedAttributesNew)
            # connect DT_i as a child node of DT
            DT.subtrees[_value] = DTchild
        return DT

    def train(self):
        start = time.time()
        self.root = self.ID3__(
            self.dataset, self.labels,
            np.arange(len(self.attributes)),
        )
        print(f"Training completed in {time.time()-start} secs")

    def predict(self, x):
        tree = self.root

        while type(tree) == TreeNode:
            _value = x[tree.attribute]
            tree = tree.subtrees[_value]

        if type(tree) == TreeLeafNode:
            return tree.label
        raise TypeError(f"Type of tree {type(tree)} is not 'TreeLeafNode'")

    def print2dotfile(self, filename="tree"):
        '''
        Writes the contents of the tree in dot file 
        format and transforms it into png using graphviz tools.
        You need graphviz to use this method (sudo apt install graphviz)
        '''
        with open(f"{filename}.dot", "w+") as f:
            self.print(file=f)
        os.system(f"dot -Tpng {filename}.dot -o {filename}.png")

    def print(self, tree=None, depth=0, file=None):
        if depth == 0:
            tree = self.root
            print("digraph dec_tree {", file=file)
        for _value in tree.subtrees.keys():
            attrName = self.attributes[tree.attribute]
            subtree = tree.subtrees[_value]
            if type(subtree) == TreeNode:
                print(
                    f'\t{attrName}_{depth} -> {self.attributes[subtree.attribute]}_{depth+1} [label = "{_value}"];',
                    file=file
                )
                self.print(subtree, depth+1, file)
            elif type(subtree) == TreeLeafNode:
                print(
                    f'\t{attrName}_{depth} -> leaf_{subtree.label}_{depth+1} [label = "{_value}"];',
                    file=file
                )
            else:
                raise TypeError(
                    f"Type of subtree {type(subtree)} is neither 'TreeNode' nor 'TreeLeafNode'")
        if depth == 0:
            print("}", file=file)
