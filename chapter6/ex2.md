# Is a nodes's Gini impurity generally lower or greater then it's parents? Is it genenerally lower/greater or alway's lower/greater.

As you down the tree nodes get purer and purer. As they contain less samples of different categories. Untill the tree is out of depth or the node is pure (and it's zero). So the node alway's has a lower gini score.

Appendix A:
Answer is wrong, you can have parents with a better gini score then one of their children. As one of the branches can just have a bit of every class. Which would result in a terrible score. It won't happen often though.
