from enum import Enum
from math import *

class NodeType(Enum):
    ADD=1
    SUB=2
    MUL=3
    COS=4
    SIN=5
    VAR=6

class Node:
    def __init__(self,node_type:NodeType):
        self._node_type=node_type
    @property
    def node_type(self):
        return self._node_type

class BinNode(Node):
    def __init__(self,node_type:NodeType,left:Node,right:Node):
        super().__init__(NodeType.ADD)
        self._left=left
        self._right=right
    @property
    def left(self):
        return self._left
    @property
    def right(self):
        return self._right

class UnNode(Node):
    def __init(self,node_type:NodeType,val):
        super().__init__(node_type)
        _val=val
    @property
    def val(self):
        return self._val

class VarNode(Node):
    def __init__(self,var_id:int):
        super().__init__(NodeType.VAR)
        self.var_id=var_id

class CosNode(UnNode):
    def __init__(self,val):
        this.val

class AddNode(BinNode):
    def __init__(self,left:Node,right:Node):
        super().__init__(NodeType.ADD,left,right)

class SubNode(BinNode):
    def __init__(self,left:Node,right:Node):
        super().__init__(NodeType.SUB,left,right)

class MulNode(Node):
    def __init__(self,left:Node,right:Node):
        super().__init__(NodeType.MUL,left,right)

def eval(node:Node,varValues):
    switcher = {
        NodeType.ADD : (lambda x: eval(x.left,varValues) + eval(x.right,varValues)),
        NodeType.SUB : (lambda x: eval(x.left,varValues) - eval(x.right,varValues)),
        NodeType.MUL : (lambda x: eval(x.left,varValues) * eval(x.right,varValues)),
        NodeType.COS : (lambda x: cos(eval(x.val,varValues))),
        NodeType.SIN : (lambda x: sin(eval(x.val,varValues))),
        NodeType.VAR : (lambda x: varValues[x.var_id]),
    }
    op = switcher.get(node.node_type,\
                      lambda x:print(str(x) + " is of invalid type"))
    return op(node)

# What is the return type of this thing?
# -> maybe array? -> but the each partial derivative needs to implement array?
# We could provide an array to put the result in? and pass this along, adding 
# the parital derivatives as we go? not sure if this makes it easier hmmm
def evalGradient(node:Node,gradValues, outputArray):
    ...

def generateGradientFunction(node:Node):
    """
    Generates a tree that represents the gradient
    """
    ...


print("running demo")

# simple example sum x1 + x2
sum = AddNode(VarNode(0),VarNode(1))
valuesSum=[3,4]
print(str(eval(sum,valuesSum)))


# example from wikipedea

#main()
