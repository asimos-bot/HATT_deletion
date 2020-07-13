# Heuristics

## Information Gain

source: https://victorzhou.com/blog/information-gain/

### Information Entropy

a dataset of the same class have zero entropy

a dataset with a mix of every class have some entropy

A good split for the tree would divide the dataset into
two nodes, each one with the same class (zero entropy).

So we just need to see the split option which has the lowest
entropy, or the one which removes more entropy from the current
dataset.

So, to every possible split we do the following calculation:

Gain = Ecurrent\_entropy - Esplit

Where Ecurrent\_entropy is the entropy of the current dataset (without split)

Where Esplit is the entropy of the split, calculate by:

Esplit = (% of current dataset in left branch) * Eleft + (% of current dataset in right branch) * Eright

Where Eleft and Eright are the entropy for each branch.

#### Calculate entropy

E = -sum( Pi * log 2 Pi )

where Pi is the probability of randomly picking an
element of class i. (the proportion of the dataset
made of class i)

Pi = (n of class i objects)/(dataset size)

## Gini Impurity

source: https://victorzhou.com/blog/gini-impurity/

This heuristic tries to answer the question: 
What's the probability we classify a random datapoint
incorrectly?

It is calculated as:

G = sum(Pi * (1 - Pi))

Where Pi is the probabilty of pickinig a datapoint with
class i

After calculating it for each branch, just like we did
in Information Gain, we weight the results by the percentage
of the current dataset that is in each branch.

Gsplit = (% percentage of current dataset in left branch) * Gleft + (% percentage of current dataset in right branch) * Gright

Where Gleft and Gright are the Gini impurity results for each branch.

All we need to do is, calculate this for every possible split
and choose the higher one.

# Heuristic Fluxogram

```
hoeffding_anytime_tree.py:

555:hoeffing_anytime_tree.partial_fit -> 603:hoeffding_anytime_tree._partial_fit -------.
							|                               |
							V                               V
				629:hoeffding_anytime_tree._process_nodes (2)   697:hoeffding_anytime_tree._sort_instance_into_leaf (1)
                                             |
					     |
					     V
				.--660:if node is AnyTimeSplitNode-----------.
                                |                                            |
			       true                                    689:false and if node is AnyTimeActiveLearningNode
				V                                            V
  .------------------ 413:AnyTimeSplitNode.learn_from_instance        862:hoeffding_anytime_tree._attempt_to_split
  |  .---- 731:hoeffding_anytime_tree._reeavaluate_best_split                                       |
  |  |      683:( go to child nodes and process them too )                                          V
  |  |                                                                               894:AnyTimeActiveLearningNode.get_best_split_suggestions
  |  `--> 773:AnyTimeSplitNode.get_best_split_suggestions                                       |
  |                                                  |                                          |
  |                                                  `---------> (calculate heuristics) <-------´
  |
  |
  `--> 443:[NominalAttributeClassObserver|NumericAttributeClassObserverGaussian].observer_attribute_class --------------.
														        |
(inner node class is AnyTimeSplitNode)                                                                                  |
(leaf node class is AnyTimeActiveLearningNode)                                                                          |
                                                                                                                        |
(calculate heuristics) ---> ( call get_best_evaluated_split_suggestion for each one in self._attribute_observers ) <----´
                                                         |
							 V
	      ( their classes are NominalAttributeClassObserver and NumericAttributeClassObserverGaussian )
                     |                                                              |
                     V                                                              V
	NominalAttributeClassObserver                               NumericAttributeClassObserverGaussian
		     |                                                              |
		     V                                                              V
     nominal_attribute_class_observer.py                             numeric_attribute_class_observer.py
                     |                                                              |
		     V                                                              V
    50:get_best_evaluated_split_criterion                            57:get_best_evaluated_split_criterion
		     |                                                              |
		     `--------------------------------.-----------------------------´
                                                      |
						      V
				   ( get_merit_of_split can be called from: )
			       InfoGainSplitCriterion:info_gain_split_criterion.py
			       GiniSplitCriterion:gini_split_criterion.py
```
