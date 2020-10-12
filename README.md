# Recursive Top-Down Production for Sentence Generation with Latent Trees

Code for the paper: http://arxiv.org/abs/2010.04704

---

To run:
```bash
python scan.py 
```
runs the code on the simple split of SCAN.

To run on the other splits, replace `<split>` with `add_jump`, `add_turn_left`, or `length`
```bash
python scan.py \
    --src_path_train data/$SPLIT/train_wo_valid_random.src \
    --trg_path_train data/$SPLIT/train_wo_valid_random.trg \
    --src_path_valid data/$SPLIT/valid.random.src \
    --trg_path_valid data/$SPLIT/valid.random.trg \
    --src_path_test  data/$SPLIT/test.src \
    --trg_path_test  data/$SPLIT/test.trg
```

---

 ## `ctreec.py`
 This is where the algorithm described in the paper is implemented in the function `forward_ctreec`.
One particular implementation detail is the order of the nodes in the array `log_probs` for `Loss`.
The order of the flattened word emission probabilities from the tree are in in-order traversal:

`ctreec_test.py` mocks a tree as shown below, and marginalises over trees of length 4.
```
              7
        /           \
       3             11       
    /     \        /     \    
   1       5      9       13   
  / \     / \    / \     /   \  
 0   2   4   6  8   10  14   15
```
Lines 77-81 in the file are the indices that the probabilities need to be extracted from to sum over
the correct trees.

For example, on line 80, a tree represented by `[3, 8, 10, 13]` would be the following:
 ```
              7
        /           \
       3             11       
                  /     \    
                 9       13   
                / \    
               8   10 
```
