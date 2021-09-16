from enum import Enum
from math import *
import numpy as np
import math


class BinType(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    COS = 4
    SIN = 5


class BinNode:
    def __init__(self, operator: BinType, left, right):
        self.operator = operator
        self.left = left
        self.right = right
        self.id = -1

    def Accept(self, visitor):
        self.left.Accept(visitor)
        self.right.Accept(visitor)
        visitor.VisitBinLate(visitor)


class UnType(Enum):
    VAR = 1
    PAR = 2  # parameter


class UnNode:
    def __init(self, node_type, val):
        super().__init__(node_type)
        self.val = val
        self.id = -1

    def Accept(self, visitor):
        self.val.Accept(visitor)
        visitor.VisitLate(visitor)


class VarNode:
    def __init__(self, var_id: int):
        super().__init__(BinType.VAR, var_id)

    def Accept(self, visitor):
        visitor.VisitVar(self)


class ParNode(UnNode):
    def __init__(self, value):
        super().__init__(BinType.PAR, value)
        self.value = value

    def Accept(self, visitor):
        visitor.VisitPar(self)


class SinNode(UnNode):
    def __init__(self, val):
        super().__init__(UnType.SIN, val)


class CosNode(UnNode):
    def __init__(self, val):
        super().__init__(UnType.COS, val)


class AddNode(BinNode):
    def __init__(self, left, right):
        super().__init__(BinType.ADD, left, right)


class SubNode(BinNode):
    def __init__(self, left, right):
        super().__init__(BinType.SUB, left, right)


class MulNode(BinNode):
    def __init__(self, left, right):
        super().__init__(BinType.MUL, left, right)


# visitor that does nothing, useful to do override with
class Visitor:
    def VisitPar(self, node: ParNode):
        return

    def VisitVar(self, node: VarNode):
        return

    def VisitUnLate(self, node: UnNode):
        return

    def VisitBinlate(self, node: UnNode):
        return


# visitor that sets the ids depth first
class SetIdVisitor(Visitor):
    def __init__(self):
        self.current = 0

    def VisitUnLate(self, node: UnNode):
        node.id = self.current
        self.current = self.current + 1
        return

    def VisitBinlate(self, node: UnNode):
        node.id = self.current
        self.current = self.current + 1
        return


def SetIds(tree):
    vis = SetIdVisitor()
    tree.Accept(vis)
    return vis.id


class EvalVisitor:
    def __init__(self, values, cache=None):
        self.values = values
        self.memory = []
        self.cache = cache

    def visitUnLate(self, node: UnNode):
        switcher = {
            BinType.COS: np.cos,
            BinType.SIN:  np.sin,
        }
        op = switcher.get(node.node_type,
                          lambda x: print(str(x) +
                                          " is invalid unitary operator"))
        innerVal = self.memory.pop()
        evaluated = op(innerVal)
        if(self.cache is not None):
            self.cache[node.id] = evaluated
        self.memory.append(evaluated)

    def visitBinLate(self, node: BinNode):
        switcher = {
            UnType.ADD: lambda l, r: l + r,
            UnType.SUB: lambda l, r: l - r,
            UnType.MUL: lambda l, r: l * r
        }
        op = switcher.get(node.node_type,
                          lambda x: print(str(x) +
                                          " is invalid binary operator"))
        # stack so fifo right then left
        right = self.memory.pop()
        left = self.memory.pop()
        evaluated = op(left, right)
        if(self.cache is not None):
            self.cache[node.id] = evaluated
        self.memory.append(evaluated)

    def visitPar(self, node: ParNode):
        self.memory.append(node.value)

    def visitVar(self, node: VarNode):
        self.memory.append(self.values[node.var_id])


def Eval(tree):
    vis = EvalVisitor()
    tree.Accept(vis)
    return tree.memory.pop()


def EvalCached(tree):
    size = SetIds(tree)
    cache = np.zeros((size, 1))
    return (Eval(tree, cache), cache)


def EvalWithCache(tree, cache):
    ...


class Gradient:
    def __init__(self, cache, dimension):
        self.cache = cache
        self.gradient = np.zeros((dimension, 1))
        self.memory_vals = []
        self.memory_indices = []

    def VisitPar(self, node: ParNode):
        return

    def VisitVar(self, node: VarNode):
        return

    def VisitUnLate(self, node: UnNode):
        return

    def VisitBinlate(self, node: UnNode):
        return



## hint: when debugging print out the seeds !!
#def evalGradient(node: Node, varValues, seed=1):
#    if(node.node_type == NodeType.ADD):
#        # derivative too left : (left+right)' == 1
#        leftSeed = seed*1
#        leftGradient = evalGradient(node.left, varValues, leftSeed)
#
#        # derivative too right : (left+right)' == 1
#        rightSeed = seed*1
#        rightGradient = evalGradient(node.right, varValues, rightSeed)
#
#        return leftGradient + rightGradient
#
#    if(node.node_type == NodeType.SUB):
#        # derivative too left : (left+right)' == 1
#        leftSeed = seed*1
#        leftGradient = evalGradient(node.left, varValues, leftSeed)
#
#        # derivative too right : (left+right)' == 1
#        rightSeed = seed*1
#        rightGradient = evalGradient(node.right, varValues, rightSeed)
#
#        return leftGradient - rightGradient
#
#    if(node.node_type == NodeType.MUL):
#        # Derivative to the left part, results in the right part.
#        leftSeed = seed*eval(node.right, varValues)
#        leftGradient = evalGradient(node.left, varValues, leftSeed)
#
#        # Derivative to the right part, results in the left part.
#        rightSeed = seed*eval(node.left, varValues)
#        rightGradient = evalGradient(node.right, varValues, rightSeed)
#
#        return leftGradient + rightGradient
#
#    if(node.node_type == NodeType.PAR):
#        return np.zeros(len(varValues))
#
#    if(node.node_type == NodeType.VAR):
#        vec = np.zeros(len(varValues))
#        vec[node.var_id] = seed
#
#        return vec
#
#    if(node.node_type == NodeType.SIN):  # -> cos(x)
#        newSeed = seed*math.cos(eval(node.val, varValues))
#
#        return evalGradient(node.val, varValues, newSeed)
#
#    if(node.node_type == NodeType.COS):  # -> cos(x)
#        newSeed = -seed*math.sin(eval(node.val, varValues))
#
#        return evalGradient(node.val, varValues, newSeed)
#
#
#print("running demo")
#
## simple example sum x1 + x2
#sum = AddNode(AddNode(VarNode(0), VarNode(1)), VarNode(0))
#valuesSum = [3, 4]
#print(str(eval(sum, valuesSum)))
#print(str(evalGradient(sum, valuesSum)))
#
#mul = MulNode(MulNode(MulNode(VarNode(0), VarNode(1)), VarNode(0)), ParNode(1))
#valuesSum = [3, 4]
#print(str(eval(mul, valuesSum)))
#print(str(evalGradient(mul, valuesSum)))
#print("theoretical answerP [24;9]")
#
## example from wiki page on autodiff
#y = MulNode(VarNode(0), VarNode(1))
#wikiExample = AddNode(y, SinNode(VarNode(0)))
#
#print("wikiexample:")
#wikiValues = [0.1, 0.3]
#print("cost: " + str(eval(wikiExample, wikiValues)))
#print("gradient with auto diff: " + str(evalGradient(wikiExample, wikiValues)))
#print("theoretical answer gradient: ["+str(
#    math.cos(wikiValues[0])+wikiValues[1]) + ";" + str(wikiValues[0]) + "]")
