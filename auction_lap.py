#!/usr/bin/env python

from __future__ import division, print_function

import sys

import torch

def auction_lap(X, eps, compute_score=True):
    """
        X: n-by-n matrix w/ integer entries
        eps: "bid size" -- smaller values means higher accuracy w/ longer runtime
    """
   
    X = torch.from_numpy(X).float()
    # --
    # Init
    
    cost     = torch.zeros((1, X.shape[1]))
    curr_ass = torch.zeros(X.shape[0]).long() - 1
    bids     = torch.zeros(X.shape)
    
    # if X.is_cuda:
    #     cost, curr_ass, bids = cost.cuda(), curr_ass.cuda(), bids.cuda()
    
    
    counter = 0
    while (curr_ass == -1).any():
        counter += 1
        
        # --
        # Bidding
        
        unassigned = (curr_ass == -1).nonzero().squeeze()
        
        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)
        
        first_idx = top_idx[:,0]
        first_value, second_value = top_value[:,0], top_value[:,1]
        
        bid_increments = first_value - second_value + eps
        
        bids_ = bids[unassigned]
        bids_.zero_()
        if unassigned.dim()==0:
            high_bids, high_bidders = bid_increments, unassigned
            cost[:,first_idx] += high_bids
            curr_ass[(curr_ass == first_idx).nonzero()] = -1
            curr_ass[high_bidders] = first_idx
        else:
            bids_.scatter_(
                dim=1,
                index=first_idx.contiguous().view(-1, 1),
                src=bid_increments.view(-1, 1)
            )
        
        
            have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()
            
            high_bids, high_bidders = bids_[:,have_bidder].max(dim=0)
            
            high_bidders = unassigned[high_bidders.squeeze()]
            
            cost[:,have_bidder] += high_bids
            
            # curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
            ind = (curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1).nonzero()
            curr_ass[ind] = -1
            
            curr_ass[high_bidders] = have_bidder.squeeze()
        
 
    score = None
    if compute_score:
        score = int(X.gather(dim=1, index=curr_ass.view(-1, 1)).sum())
    
    return score, curr_ass.numpy(), counter