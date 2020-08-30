from enum import Enum
from math import *
import numpy as np
import math

class NodeType(Enum):
    ADD=1
    SUB=2
    MUL=3
    COS=4
    SIN=5
    VAR=6
    PAR=7

class Node:
    def __init__(self,node_type:NodeType):
        self.node_type=node_type

class BinNode(Node):
    def __init__(self,node_type:NodeType,left:Node,right:Node):
        super().__init__(node_type)
        self.left=left
        self.right=right

class UnNode(Node):
    def __init(self,node_type:NodeType,val):
        super().__init__(node_type)
        self.val=val

class VarNode(Node):
    def __init__(self,var_id:int):
        super().__init__(NodeType.VAR)
        self.var_id=var_id

class ParNode(Node):
    def __init__(self,value):
        super().__init__(NodeType.PAR)
        self.value = value

class SinNode(Node):
    def __init__(self,val):
        super().__init__(NodeType.SIN)
        self.val = val

class CosNode(Node):
    def __init__(self,val):
        super().__init__(NodeType.COS)
        self.val = val

class AddNode(BinNode):
    def __init__(self,left:Node,right:Node):
        super().__init__(NodeType.ADD,left,right)

class SubNode(BinNode):
    def __init__(self,left:Node,right:Node):
        super().__init__(NodeType.SUB,left,right)

class MulNode(BinNode):
    def __init__(self,left:Node,right:Node):
        super().__init__(NodeType.MUL,left,right)

def createSingle(varValues,indexValue, indexVector):
    #assert(indexValue < len(varValues)
    vec = np.zeros(len(varValues))
    vec[indexVector] = varValues[indexValue]
    return vec

def eval(node:Node,varValues):
    switcher = {
        NodeType.ADD : (lambda x: eval(x.left,varValues) + eval(x.right,varValues)),
        NodeType.SUB : (lambda x: eval(x.left,varValues) - eval(x.right,varValues)),
        NodeType.MUL : (lambda x: eval(x.left,varValues) * eval(x.right,varValues)),
        NodeType.COS : (lambda x: np.cos(eval(x.val,varValues))),
        NodeType.SIN : (lambda x: np.sin(eval(x.val,varValues))),
        NodeType.VAR : (lambda x: varValues[x.var_id]),
        NodeType.PAR : (lambda x: x.value)
    }
    op = switcher.get(node.node_type,\
                      lambda x:print(str(x) + " is of invalid type"))
    return op(node)

def deps(node:Node):
    """
    Find the dependency of the expressions
    """
    switcher = {
        NodeType.ADD : (lambda x: deps(x.left)  + deps(x.right)),
        NodeType.SUB : (lambda x: deps(x.left)  + deps(x.right)),
        NodeType.MUL : (lambda x: deps(x.left)  + deps(x.right)),
        NodeType.COS : (lambda x: deps(x.val)),
        NodeType.SIN : (lambda x: deps(x.val)),
        NodeType.VAR : (lambda x: [x.var_id]),
        NodeType.PAR : (lambda x: [])
    }
    op = switcher.get(node.node_type,\
                      lambda x:print(str(x) + " is of invalid type"))
    return op(node)

def createUnit(dimension,index):
    vec = np.zeros(dimension)
    vec[index] = 1
    return vec

# hint: when debugging print out the seeds !!
def evalGradient(node:Node,varValues,seed=1):
    if(node.node_type==NodeType.ADD):
        # derivative too left : (left+right)' == 1
        leftSeed = seed*1
        leftGradient = evalGradient(node.left,varValues,leftSeed) 

        # derivative too right : (left+right)' == 1
        rightSeed = seed*1
        rightGradient = evalGradient(node.right,varValues,rightSeed)

        return leftGradient + rightGradient

    if(node.node_type==NodeType.SUB):
        # derivative too left : (left+right)' == 1
        leftSeed = seed*1
        leftGradient = evalGradient(node.left,varValues,leftSeed) 

        # derivative too right : (left+right)' == 1
        rightSeed = seed*1
        rightGradient = evalGradient(node.right,varValues,rightSeed)

        return leftGradient - rightGradient

    if(node.node_type==NodeType.MUL):
        # Derivative to the left part, results in the right part.
        leftSeed = seed*eval(node.right,varValues)
        leftGradient = evalGradient(node.left,varValues,leftSeed) 

        # Derivative to the right part, results in the left part.
        rightSeed = seed*eval(node.left,varValues)
        rightGradient = evalGradient(node.right,varValues,rightSeed)

        return leftGradient + rightGradient

    if(node.node_type==NodeType.PAR):
        return np.zeros(len(varValues))

    if(node.node_type==NodeType.VAR):
        vec = np.zeros(len(varValues))
        vec[node.var_id] = seed

        return vec

    if(node.node_type==NodeType.SIN): # -> cos(x)
        newSeed = seed*math.cos(eval(node.val,varValues))
        
        return evalGradient(node.val,varValues,newSeed)

    if(node.node_type==NodeType.COS): # -> cos(x)
        newSeed = -seed*math.sin(eval(node.val,varValues))
        
        return evalGradient(node.val,varValues,newSeed)

print("running demo")

# simple example sum x1 + x2
sum = AddNode(AddNode(VarNode(0),VarNode(1)),VarNode(0))
valuesSum=[3,4]
print(str(eval(sum,valuesSum)))
print(str(evalGradient(sum,valuesSum)))

mul = MulNode(MulNode(MulNode(VarNode(0),VarNode(1)),VarNode(0)),ParNode(1))
valuesSum=[3,4]
print(str(eval(mul,valuesSum)))
print(str(evalGradient(mul,valuesSum)))
print("theoretical answerP [24;9]")

# example from wiki page on autodiff
y = MulNode(VarNode(0),VarNode(1))
wikiExample = AddNode(y,SinNode(VarNode(0)))

print("wikiexample:")
wikiValues = [0.1,0.3]
print("cost: " + str(eval(wikiExample,wikiValues)))
print("gradient with auto diff: " + str(evalGradient(wikiExample,wikiValues)))
print("theoretical answer gradient: ["+str(math.cos(wikiValues[0])+wikiValues[1]) + ";" + str(wikiValues[0]) + "]")
