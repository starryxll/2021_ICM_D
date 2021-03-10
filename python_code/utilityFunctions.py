# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:12:10 2021

@author: haochen
"""
import GraphDataStructure
import numpy as np
import random
from autograd import grad
import math
def constructGraph(my_data, my_graph):  # input is numpy arrays and out put is a Graph object
    # my_graph = Graph()
    my_edge = Edge()
    edge = []
    vertex = []
    for i in range(len(my_data)):
        left = my_data[i][0]
        right = my_data[i][1]
        mygraph.edgeList.append(Edge(left, right))
        if left not in vertex:
            my_graph.verticesL.append(left)
    my_graph.numVertices = len(vertex)
    my_graph.numEdges = len(edge)
    my_graph.constEdgeByVertex()      # find output edge for every vertex
    
def motifsEnumeration(graph):
    '''
    Based on ESE(exact subgraph enumeration algorithms from http://snap.stanford.edu/class/cs224w-2019/slides/03-motifs.pdf p34)
    to go through the entire graph and find subgraphs corresponding to every 3-node motifs
    
    Input: Graph data structure
    Output: count for every 3-node motifs in the Graph
    '''
    motifCounts = [0]*13  # there are 13 kinds of motifs for 3-node subgraph, 
    for vertex in graph:
        extendSubgraph([], graph, vertex, motifCounts) # at the begining, the subgraph is empty and the extension graph is the entire graph
    
def extendSubgraph(subgraph, extension, vertex,motifCounts):
    if len(subgraph)==3: # now,subgraph contains 3 vertices
        motifCounts[motifMatch(edges)]+=1
    while len(extension!=0):
        if (len(subgraph==1)):
            subNeighbor = subgraph[0].outputEdgesByVertex
        else:
            subNeighbor = subgraph[0].outputEdgesByVertex.append(subNeighbor = subgraph[1].outputEdgesByVertex)
        for tempVertex in extension:  # find the first vertex in extension graph that is neighbor of the current subgraph
            if tempVertex in subNeighbor:
                subgraph.append(tempVertex)
                extension.remove(tempVertex)
                continue
        extendSubgraph(subgraph, extension, vertex, motifCounts) # recursive go through the entire graph from the input starting point "vertex"
    
def motifMatch(edges):
    '''
    Input: 3 dimension vector representing the edges for 3 node, 0 means no link betetween
            2 nodes, 1 means from the left node to the right, 2 means from the right node to the left
            3 means bidirectional link between this two nodes
    
    Output: scalar value in range(13) corresponding to the 3-node motifs,
            for more details, see see http://snap.stanford.edu/class/cs224w-2019/slides/03-motifs.pdf p4
    '''
    value = edges[0]*100+edges[1]*10+edges[2])
    
    # if(value=220):
    #     return 0
    # else if(value=230):
    #     return 1
    # else if(value=240):
    #     return 2
    # else if(value=203):
    #     return 3
    # else if(value=223):
    #     return 4
    # else if(value=224):
    #     return 5
    # else if(value=340):
    #     return 6
    # else if(value=440):
    #     return 7
    # else if(value=322):
    #     return 8
    # else if(value=422):
    #     return 9
    # else if(value=432):
    #     return 10
    # else if(value=442):
    #     return 11
    # else if(value=444):
    #     return 12
    
    matching_dict = {220:0,230:1,240:2,203:3,223:4,224:5,340:6,440:7,322:8,422:9,432:10,442:11,444:12}
    return matching_dict[value]

## utility functions for problem (4)      
def findCommunity(graph):
    '''
    Based on Louvian Algorithm from http://snap.stanford.edu/class/cs224w-2019/slides/04-communities.pdf p34,
    but optimized much more with the devide and conquer, so that we can run much faster
    
    '''
    community = copy.deepcopy(graph)
    # initially, each node is a community
    merge_length = int(len(graph)/2)*2     # if the the graph length is odd, then minus one
        # make sure that merge_length is always even number
        
        #### devide and conquer begins
    while(merge_length>1):
        
        # make sure that length of community is always even
        if len(graph)/2 != 0:    # the first on should be merged into the second one
            merge_length = int(merge_length/2)
            leftGraph = community.remove(community[0])
            community.remove(leftGraph)
            rightGraph = community.remove(community[0])     
            community.append(leftGraph.append(rightGraph))  
    
        for i in merge_length/2:   
            merge_length = int(merge_length/2)
            leftGraph = community.remove(community[0])
            community.remove(leftGraph)
            rightGraph = community.remove(community[0])      
            community.append(mergeCommunity(leftGraph, rightGraph))        
    #### devide and conquer ends
    
    return commutiny

def mergeCommunity(leftGraph, rightGraph):
    #### phase 1, find the vertex that maximize the community gain
    for tempVertex in leftGraph:
        if computeModularityGain(leftGraph.append(rightGraph), leftGraph, tempVertex):
            #### phase 2, rebuild the graph
            return leftGraph.append(rightGraph)
    return [leftGraph, rightGraph]
    
def computeModularityGain(totalGraph, graph, vertex):
    m = len(graph)
    summationIn = 0
    summationTotal= sum(graph.inputEdgesByVertex.edgeList.weight) 
    summationVertex = sum(vertex.outputEdges.weight)
    summationVG = 0
    for tempVertex in graph:
        if [tempVertex, vertex] in totalGraph.edgeList:
            summationVG += 1
        for tempVertex2 in graph:
            if tempVertex==tempVertex2:
                continue
            if [tempVertex, tempVertex2] in totalGraph.edgeList:
                summationIn += 1
    return ((summationIn+summationVertex)/(2*m)-power(summationVG+summationTotal,2)/(2*m)) -(summationIn/(2*m) - power(summationTotal/(2*m), 2) - power(summationVG/(2*m),2)) > 0       
  
                
## utility functions for problem (5)
#### use EM algorithm to run mcmc(E step) and parameterLearning(M step) interchangeably 
#### until converge (the new mcmc step output samples that should be similar enough to last mcmc step)
def em(graph, startingVert):
    # randomly initialize the MCMC samledata
    sampleData = list(random.sample(range(graph.vertList.shape[0]),size=5)) * graph.vertList.shape[1]
    
    while(True):
        # update graph weight by parameterl learing
        graph.vertList.weight = parameterLearning(graph, sampledata)
    
        # generate new mcmc samples with the updated weight
        newData = mcmc(graph, startingVert)
        
        if(shouldBeStop(sampleData, newData, lastDifference)):
            break
        
def shouldBeStop(old, new, lastDifference):
    # both old and new are matrix and share the same structure, and lastDifference is a scalar
    newDifference=sum(sum(old-new))/(old.shape[0]*old.shape[1])        
    if (lastDifference-newDifference/lastDifference) > 0.99:
        return false # should stop em algorithm
    else:
        return true
    # don't forget to update the sampleData
    old = new 
    
def parameterLearning(graph, data):
    '''
    Input: graph structure with vertices and edges, and musician based features
    
    Output: the weighting parameters for each music features
    
    Assumption: during MCMC walking, we assume the binary distribution when going from one node to another node,
                and the probability is the ration computed by out degree of starting node
                
                we take step length to be 5, sample size 1000 for each starting vertex
    '''
    
    # step-1 use negative sampling on the data to compute the likelihood gradient for the graph Markov chain 
    # sample
    random.shuffle(graph.vertList.features)
    featureMatrix=graph.vertList.features[0:len(graph.vertList.features)/2]
    # gradient with negative sampling
    gradient = 1 / (1+ math.exp(-(np.dot(graph.vertList.features,graph.featureWeight) - np.dot(featureMatrix, graph.featureWeight))))
    
    # take learning rate to be 0.01
    graph.vertList.features -= 0.01*gradient
    
    
    # step-2 use gradient descent to update the weight parameter 
    
def mcmc(graph, startingVert):
    # note that startVert is a vector
    path = []
    sample=[]
    # note that the sample should be used for two parts, first for the learning process, 
    # second is for the dynamic influence analysis
    randomMatrix = np.random.rand(len(startingVert),5)
    for i in range(1000):
        path = []
        for j in range(5):
            # weightList is a 2d matrix
            weightList = startingVert.outputVertex.weight / sum(startingVert.outputVertex.weight)
            for i in range(len(weightList)==1):
                weightList[i+1,:] = weightList[i,:]+weightList[i+1,:]
                nextIndex[i] = findNextStepIndex(randomMatrix, weightList, j)
            nextStep=startingVert.outputEdges[nextIndex]
            path=zip(startingVert, nextStep)
            startingVert = nextStep
        sample.append(path)
        
def findNextStepIndex(randomMatrix, weightList, j):
    if not (j<5):
        raise AssertionError("the index for step should be smaller than 5")
    indexArray=[]
    for i in range(len(weightList)):
        if randomMatrix[i,j]<weightList[i+1] & randomMatrix[i,j]>weightList[i]:
            indexArray[j] = i
            
def outbreakDetection(dynamicGraph):
    """
    Based on backpropagation, we first delete last significant edges that are in 2010, then delete edges in 2020,
    so on until detele edges in 1930, the remain vertices(mucisians) should be the source of outbreak
    
    Besides, duting the dynamic detection, we store the delete process to represent the evolution process which can be 
    sample data used in dynamicAnalysis
    """ 
    
    # it's better the put the influencer and samplePath change inside the object and store it 
    for vertex in dynamicGraph.vertList:
        myGraph = copy.deepcopy(dynamicGraph) # must not alter the data store in the dynamicGraph
        backpropDelete(myGraph, vertex) 
    
    
def backpropDelete(myGraph, vertex):
    
    """
    Recursively find influencer and corresponding samplePath
    """
    while (not isempty(myGraph)):
        for neighbor in vertex.outputEdgesByVertex:
            backpropDelete(myGraph, neighbor[1])
        samplePath=[]
        for i in range(len(myGraph.timeLine), -1, -1):
            tempTime = timeLine[i]
            N = len(myGraph.outputDegree)
            tempAvg = sum(myGraph.outputDegree)/N
            for vertex in myGraph.vertList:
                if(vertex.outputDegree<tempAvg):
                    myGraph.vertList.remove(vertex)
                    samplePath.append(vertex)
        influencer = ""
        tempDegree = 0
        for vertex in myGraph.vertList:
            if vertex.outputDegree > tempDegree:
                influencer = vertex
                tempDegree = influencer.outputDegree
    dynamicGraph.influencers.append(influencer)
    dynamicGraph.samplePath.append(samplePath)
    
def dynamicAnalysis(dynamicGraph, influencers, samplePath, weigthParameter):
    '''
    the samplePath is computed from the ourbreakDetection, and other inputs are already computed in other methods

    '''
    # note that the influencers is vertices in the dynamicGrpah(subgraph of the total graph)
    # in this function, we want to draw a  plot containing several lines to demonstrate the 
    # change pattern for different genres and different artists
   
    # The information for genre change pattern can be gather from vertex features in a subgraph,
    # and the artist change patten(several important vertices, not all vertices) can be find besed on single vertex.
    
    # both plots are based on one metric(weighted output degree)

def computeMetric(elements, weights):
    # elements can be a genre(subgraph, sevel vertices) or a single musician(single vertices)
    if isinstance(elements, Graph) or isinstance(elements, Vertex):
        return np.dot(elements.features, weights) * elements.outputDegree
    else:
        raise Exception("should use Graph or Vertex datastructure to compute the metrics")
    