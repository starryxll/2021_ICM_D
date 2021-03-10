# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:15:01 2021

@author: haochen
"""
        
class Vertex(object):
    """
    Vertex object
    """
    def __init__(self, key):
        self.key = key
        self.connectedTo = {}    # store another vertex that this vertex connects to
        self.outputDegree = 0   
        
    def addNeighbor(self, nbr, weight):
        self.connectedTo.update({nbr: weight})

    def __str__(self):
        return str(self.key) + '-->' + str([nbr.key for nbr in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.key

    def getWeight(self, nbr):
        weight = self.connectedTo.get(nbr)
        if weight is not None:
            return weight
        else:
            raise KeyError("No such nbr exist!")

class Egde(object):     
    """
    Edge object
    The dataType for Edge object is (leftKey, rightKey) a.k.a (string, string)
    """
    def __init__(self, leftVert, rightVert):
        self.weight = 1
        self.connectedVertices = [leftVert, rightVert]
                
        
class Graph(object):
    """
    Directed graph object
    """
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0
        self.edgeList = {}
        self.numEdges = 0
        self.outputEdgesByVertex = [[]]  # 2d arrays
        self.inputEdgesByVertex = [[]]   # 2d arrays
        self.featureWeight = []
        
    def addVertex(self, key):
        self.vertexList.update({key: Vertex(key)})
        self.vertexNum += 1

    def getVertex(self, key):
        vertex = self.vertexList.get(key)
        return vertex

    def __contains__(self, key):
        return key in self.vertexList.values()

    def addEdge(self, f, t, weight=0):
        f, t = self.getVertex(f), self.getVertex(t)
        if not f:
            self.addVertex(f)
        if not t:
            self.addVertex(t)
        f.addNeighbor(t, weight)

    def getVertices(self):
        return self.vertexList.keys()
    
    def getEdgeByVertex(vertexKey):
        return self.edgeList.()

    def __iter__(self):
        return iter(self.vertexList.values())
    
    def constEdgeByVertex(self.vertList, self.edgeList):
        for tempVertex in self.VertList:
            tempEdgeListOut = []
            tempEdgeListIn = []
            for tempEdge in self.edgeList:
                if tempEdge[0] = tempVertex:
                    tempEdgeListOut.append(tempEdge)
                if tempEdge[1] = tempVertex:
                    tempEdgeListIn.append(tempEdge)
            self.outputEdgesByVertex.append(tempEdgeListOut)
            self.inputEdgesByVertex.append(tempEdgeListIn)
        
class DynammicGraph(Graph):
    """
    This graph is constrcuted based on time line, so for each time point, there should be 
    one corresponding subgraphs, and each vertex (representing a musian or a music) contains features as vectors
    """
    def __init__(self):
        self.timeLine = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000. 2010]
        self.vertList = {1930:[], 1940:[], 1950:[], 1960:[], 1970:[], 1980:[], 1990:[], 2000:[], 2010:[]}
        self.numVertices = 0
        self.edgeList = {1930:[], 1940:[], 1950:[], 1960:[], 1970:[], 1980:[], 1990:[], 2000:[], 2010:[]}
        self.numEdges = 0
        self.outputEdgesByVertex = {1930:[], 1940:[], 1950:[], 1960:[], 1970:[], 1980:[], 1990:[], 2000:[], 2010:[]}  # 2d arrays
        self.inputEdgesByVertex = {1930:[], 1940:[], 1950:[], 1960:[], 1970:[], 1980:[], 1990:[], 2000:[], 2010:[]}   # 2d arrays
        self.features = []
        self.influencers = influencers
        self.samplePath = samplePath
    
    def setFeatures(data):
        self.features = data
        
    # can use methods in utilityFunctions to construct edges and vertex for the DynamicGraph
    

    