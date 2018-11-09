def left_stripper(st, char = '('):
    li = []      
    i = 0   
    if st[i] == char:
        while st[i] == char:
            li .append(st[i])        
            i += 1
    else:
        li.append(st)   
    return li




def right_stripper(st, char = ')'):
    li = []  
    i = len(st) - 1    
    while st[i] == char:
        li .append(st[i])        
        i -= 1
    li.insert(0,st[:i+1])
    return li





def sentence_parser(st):   
    li = st.strip().split()
    #li = [right_stripper(tok) for tok in li]
    #li = [item for sublist in li for item in sublist]
    #li = [left_stripper(tok) for tok in li]
    #li = [item for sublist in li for item in sublist]
    return li
    #return li[1:-1]   #drop the outermost brackets



class Node:  # a node in the tree
    def __init__(self, word=None):        
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child        
        self.isLeaf = False      



class Tree:

    def __init__(self, treeString, openChar='(', closeChar=')'):
        tokens = []
        self.open = openChar
        self.close = closeChar
        tokens = sentence_parser(treeString)
        self.root = self.parse(tokens)
        self.binarize(self.root)
        self.string = self.stringify(self.root)
        self.tokens = self.recover_sentence(self.root)
        self.transitions = self.transition_parser()
    
    def recover_sentence(self, node):
        sentence_list = []       
        if node.isLeaf:
            sentence_list.append(node.word)
            return sentence_list
        else:
            left_branch = self.recover_sentence(node.left)          
            right_branch = self.recover_sentence(node.right)           
            return left_branch + right_branch
            
    def transition_parser(self):
        tokens = self.string.split()
        trans = []
        for tok in tokens:
            if tok == ")" :
                trans.append(2)
            elif tok == "(":
                pass
            else:
                trans.append(1)
        return trans
        
    
    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"
        
        split = 1  # position after open 
        countOpen = countClose = 0
        
        if tokens[split] == self.open:
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1
        
        # New node
        node = Node()  
        node.parent = parent
        
        if countOpen == 0:
            node.word = ''.join(tokens[1:-1]).lower()  # lower case?
            node.isLeaf = True
            return node
        
        node.left = self.parse(tokens[1:split], parent=node)
        if tokens[split] == self.open:
            node.right = self.parse(tokens[split:-1], parent=node)
        else:
            node.right = None             
        return node
    
    
    def fuse(self, mother, child):
        if mother == self.root:
            self.root = child
            return
        else:
            forebear = mother.parent
            child.parent = forebear
            if forebear.left == mother:
                forebear.left = child
            else:
                forebear.right = child
    
    def stringify(self, node):    
        if node == None:
            return ""
        elif node.isLeaf:      
            return  node.word 
        else:           
            return "( " + self.stringify(node.left) + " " + self.stringify(node.right) + " )"
            
        
    def is_binary(self, node):
        if node.isLeaf:
            return True
        elif ((node.right == None) or (node.left == None)):            
            return False
        else:
            check1 = self.is_binary(node.left)
            check2 = self.is_binary(node.right)
            return (check1 and check2)
        
    def is_binary_tree(self):
        return self.is_binary(self.root)
        
    def is_binary_node(self, node):
        if node.isLeaf:
            return True
        elif ((node.right == None) or (node.left == None)):
            return False
        else:
            return True
            
    def binarize(self, node):
        if node.isLeaf:
            return
        if self.is_binary_node(node):
            self.binarize(node.left)
            self.binarize(node.right)            
        else:
            if (node.right == None):
                child = node.left
            else:
                child = node.right
            self.fuse(node, child)
            self.binarize(child)          


