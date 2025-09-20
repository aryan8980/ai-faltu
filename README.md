# ai-faltu

prac 1 ( n-queen)
"n_queen problem (chess)"
q= int(input("enter the number of queen: "))
board = [[0]*q for i in range(q)]

def is_attack(i,j):
    for k in range(q):
        if board[i][k]==1 or  board[k][j]==1:
            return True
        for k in range(0,q):
            for l in range(0,q):
                if (k+l==i+j) or (k-l==i-j):
                    if(board[k][l]==1):
                        return True
    return False

def n_queen(n):
    if n==0:
        return True
    for i in range(0,q):
        for j in range(0,q):
            if(not(is_attack(i,j))) and (board[i][j]!=1):
                board[i][j]=1
                if n_queen(n-1)==True:
                    return True
                board[i][j]=0
    return False

if n_queen(q):
    for row in board:
        print(row)
else:
    print("wrong solution")


**prac 1 ( magic sqaure)**

"magic square"
magic_square=[[0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0]]

n=5
num=1
row=0
col=n//2

while num<=n*n:
    magic_square[row][col]=num
    num+=1

    prev_row=row
    prev_col=col

    row-=1
    col+=1

    if row<0:
        row=n-1
    if col==n:
        col=0

    if magic_square[row][col]!=0:
        row=prev_row+1
        col=prev_col
        if row==n:
            row=0
    
print("Magic Square:")
for r in magic_square:
    print(r)


#3x3 magic square

magic_square = [[0,0,0],
                [0,0,0],
                [0,0,0]]
n = 3
num = 1
row = 0
col = n // 2

while num <=n*n:
    magic_square[row][col]=num
    num+=1

    prev_row=row
    prev_col=col

    row-=1
    col+=1
    if row<0:
        row=n-1
    if col==n:
        col=0
    if magic_square[row][col]!=0:
        row=prev_row+1
        col=prev_col
        if row==n:
            row=0
print("3x3 Magic Square:")
for r in magic_square:
    print(r)
    

**bfs sequence (prac 2)**

"bfs sequence (breadth first search)"
from collections import deque
def bfs_seq(graph,start):
    queue= deque([start])
    visited=set()
    path=[]

    while queue:
        vertex=queue.popleft()
        if vertex not in visited:
            path.append(vertex)
            visited.add(vertex)
            for neighbor in graph.get(vertex,[]):
                queue.append(neighbor)
    return path

if __name__=="__main__":
    example_graph={
        'a':['b','c'],
        'b':['d','e'],
        'c':['g','f'],
        'd':[],
        'e':[],
        'f':[],
    }

start_node='a'
bfs_se= bfs_seq(example_graph,start_node)
print(f"path from {start_node}: {bfs_se}")

**dfs algo (prac 2)**

def dfs(graph, start):
    stack = [start]
    visited = set()
    path = []

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            path.append(vertex)
            visited.add(vertex)
            for neighbor in reversed(graph.get(vertex, [])):
                stack.append(neighbor)
    return path

ex_graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

start_node = 'A'
result_path = dfs(ex_graph, start_node)
if result_path:
    print(f"Path from {start_node} : {result_path}")
else:
    print(f"No path found from {start_node} : {result_path}")


**a star algo (prac 3)**
"heuristic search "
"step 1- create a heuristic function"
"step 2- create a graph"
"step 3- implement a* algorithm"
"step 4- get neighbors of a node"
"step 5- call the a* algorithm"
"step 6- print the path"


def heuristic(n):
    H_dist={
        'a':1,'b':3,'c':3,'d':2,'e':4,'f':2,'g':3,'h':4,
    }
    return H_dist.get(n,0)

graph_nodes={
    'a':[('b',1),( 'c',4)],
    'b':[('d',2),('e',5)],
    'c':[('f',1),('g',3)],
    'd':[],
    'e':[('h',2)],
    'f':[],
    'g':[],
    'h':[],
}
def a_algo(start,goal):
    open_set=set(start)
    closed_set=set()
    g={}
    parents={}
    g[start]=0
    parents[start]=start
    while len(open_set)>0:
        n=None
        for v in open_set:
            if n==None or g[v]+heuristic(v)<g[n]<g[n]+heuristic(n):
                n=v
        if n==goal or graph_nodes[n]==None:
            pass
        else:
            for (m,weight) in graph_nodes[n]:
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m]=n
                    g[m]=g[n]+weight
                else:
                    if g[m]>g[n]+weight:
                        g[m]=g[n]+weight
                        parents[m]=n
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)
        if n==None:
            print("Path does not exist")
            return None
        if n==goal:
            path=[]
            while parents[n]!=n:
                path.append(n)
                n=parents[n]
            path.append(start)
            path.reverse()
            print("Path found:{}".format(path))
            return path
        open_set.remove(n)
        closed_set.add(n)
    print("Path does not exist")
    return None
def get_neighbors(v):
    if v in graph_nodes:
        return graph_nodes[v]
    else:
        return None
a_algo('a','h')

**log function minimax(prac 4)**
"log function minimax function"
import math
def minMax(cd,node,maxt,src,td):
    if cd ==td:
        return src[node]
    if maxt:
        return max(minMax(cd+1,node*2,False,src,td),
                   minMax(cd+1,node*2+1,False,src,td))
    else:
        return min(minMax(cd+1,node*2,True,src,td),
                   minMax(cd+1,node*2+1,True,src,td))
    
src=[]
x=int(input("enter the number of lead node: "))
for i in range(x):
    y=int(input("enter the value of lead node: "))
    src.append(y)
td = int(math.log(len(src),2))
cd = int(input("enter current depth: "))
nodeV= int(input("enter the node value: "))
maxT = True
print("the ans is: ",end='')
Answer = minMax(cd,nodeV,maxT,src,td)
print(Answer)

**alpha beta (prac 4)**

import math

def alpha_beta(depth, node_index, is_max, values, alpha, beta, max_depth):
    
    if depth == max_depth:
        return values[node_index]

    if is_max:  
        best = -math.inf
        for i in range(2):  
            val = alpha_beta(depth + 1, node_index * 2 + i, False, values, alpha, beta, max_depth)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:  
                break
        return best
    else:  #
        best = math.inf
        for i in range(2):
            val = alpha_beta(depth + 1, node_index * 2 + i, True, values, alpha, beta, max_depth)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:  
                break
        return best


values = [2, 8, 3, 1, 6, 9, 8, 9, 3, 10, 2, 14, 16, 8, 15, 18]

max_depth = int(math.log(len(values), 2))


alpha = -math.inf
beta = math.inf


result = alpha_beta(0, 0, True, values, alpha, beta, max_depth)
print("The optimal value is:", result)



**fuzzy set union intersection(prac 5)**
"Fuzzy set union and compliment relation"

A=dict()
B=dict()
AuniB=dict()
AintB=dict()
Acom=dict()

A={"x1":0.5,"x2":0.7,"x3":0}
B={"x1":0.8,"x2":0.2,"x3":1}

print("Fuzzy set A: ",A)
print("Fuzzy set B: ",B)

for A_key, B_key in zip(A,B):
    A_value=A[A_key]
    B_value=B[B_key]
    if A_value>B_value:
        AuniB[A_key]=A_value
        AintB[B_key]=B_value
    else:
        AuniB[B_key]=B_value
        AintB[A_key]=A_value
print("Union of A and B: ",AuniB)
print("Intersection of A and B: ",AintB)

**min-max composition (prac6)**
"min-max composition"

import numpy as np
def maxMin(x, y):
    z = []
    for x1 in x:
        row = []
        for col in range(y.shape[1]):
            min_list = [min(x1[k], y[k][col]) for k in range(len(x1))]
            row.append(max(min_list))
        z.append(row)
    return np.array(z)
r1 = np.array([[0.2, 0.5], [0.7, 0.9]])
r2 = np.array([[0.4, 0.3], [0.6, 0.1]])
print(maxMin(r1, r2))

**max-product composition (prac 6)**


"max-product composition"

import numpy as np

def maxProduct(x,y):
    z=[]
    for x1 in x:
        for y1 in y.T:
            z.append(max(np.multiply(x1,y1)))
    return np.array(z).reshape((x.shape[0],y.shape[1])) 

r1= np.array(([0.2,0.5],[0.7,0.9]))
r2=np.array(([0.4,0.3],[0.6,0.1]))


print(maxProduct(r1,r2))

**max-average composition(prac 6)**

"max-average composition"

import numpy as np

def maxAvg(x,y):
    z=[]
    for x1 in x:
        for y1 in y.T:
            z.append(1/2*max(np.add(x1,y1)))
    return np.array(z).reshape((x.shape[0],y.shape[1]))

r1= np.array(([0.2,0.5],[0.7,0.9]))
r2=np.array(([0.4,0.3],[0.6,0.1]))
print(str(maxAvg(r1,r2)))

**Matlab Code: (prac 7,8)**

fuzzy controller washing machine: 
w = readfis("mamdanitype1.fis"); 
dirt=input("provide dirt level in percentage: "); 
grease=input("provide grease level in percentage: "); 
o=evalfis([dirt, grease], w) 
disp=(["time to wash the clothes: " num2str(o)]) 

cruise controller: 
q = readfis("mamdanitype2.fis"); 
speed=input("provide speed: "); 
distance=input("provide distance: "); 
o=evalfis([distance, speed], q) 
disp=(["answer is: " num2str(0)])

**NLTK (prac 10):**

just remember this code dont do anything:

1.
 import nltk
 nltk.download("wordnet")
 from nltk.stem import WordNetLemmatizer
 from nltk.corpus import wordnet
 lemmatizer = WordNetLemmatizer()
 # reduce a word to its base or dictionary form
 print('rocks: ', lemmatizer.lemmatize('rocks'))
 print('corpora: ', lemmatizer.lemmatize('corpora'))
 print('better: ', lemmatizer.lemmatize('better', pos='a'))  # adjective
 print('beautiful: ', lemmatizer.lemmatize('beautiful', pos='a'))
 print('worst: ', lemmatizer.lemmatize('worst', pos='a'))
 print('eating: ', lemmatizer.lemmatize('eating', pos='v'))
 print('amazing: ', lemmatizer.lemmatize('amazing', pos='a'))
 print('beauty: ', lemmatizer.lemmatize('beauty', pos='n'))
 # Lemmatizing 'running' as a verb
 lemma_verb = lemmatizer.lemmatize("running", pos=wordnet.VERB)
 print(f"Lemma of 'running' as a verb: {lemma_verb}")
 # Lemmatizing 'running' as a noun (default behavior)
 lemma_noun = lemmatizer.lemmatize("running")
 print(f"Lemma of 'running' as a noun: {lemma_noun}")
 # Lemmatizing 'better' as an adjective
 lemma_adj = lemmatizer.lemmatize("better", pos=wordnet.ADJ)
 print(f"Lemma of 'better' as an adjective: {lemma_adj}")


2.
  import nltk
 nltk.download("punkt_tab")  # pre-trained, unsupervised trainable tokenizer
 from nltk.stem import PorterStemmer
 from nltk.tokenize import sent_tokenize, word_tokenize
 # Example sentence
 sentence = "Pollution is the introduction of harmful materials, called pollution, into the 
environment"
 # Word tokenization
 words = word_tokenize(sentence)
 print("Word Tokens: ", words)
 # Sentence tokenization
 sent_tokens = sent_tokenize(sentence)
 print("Sentence Tokens: ", sent_tokens)
 #  Stemming using PorterStemmer
 ps = PorterStemmer()
 for w in words:
    rootWord = ps.stem(w)
    print(rootWord)


3.
 from nltk.tokenize import RegexpTokenizer
 from collections import Counter
 # it will extract sequences of one or more word characters as individual tokens
 # effectively separates words and numbers while omitting punctuation
 tokenizer = RegexpTokenizer(r'\w+')  # raw string, word character
 text = "Pollution is the process of making the environment pollute the water and the air 
by adding harmful substances. Pollution causes an imbalance."
 # Tokenize the text
 tokens = tokenizer.tokenize(text)
 print("Tokens:", tokens)
 # Filter only alphanumeric tokens
 filtered = [w for w in tokens if w.isalnum()]
 # Count word frequencies
 counts = Counter(filtered)
 print("Word Frequency:", counts)


 **Neural Network(prac 11)**


 from numpy import random, exp, array, dot
 from collections import defaultdict, deque
 training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1]])
 training_set_outputs = array([[0, 1, 1]]).T
 random.seed(1)
 synaptics_weights = 2 * random.random((3, 1)) - 1
 print(synaptics_weights)

 for i in range(10000):
    output = 1 / (1 + exp(-dot(training_set_inputs, synaptics_weights)))
    synaptics_weights += dot(training_set_inputs.T,
                             (training_set_outputs - output) * output * (1 - 
output))
 print(synaptics_weights)
 print(1/(1+exp(-(dot(array([0,1,1]),synaptics_weights)))))
