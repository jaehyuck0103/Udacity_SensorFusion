/* \author Aaron Brown */
// Quiz on implementing kd tree

#pragma once
#include "../../render/render.h"

// Structure to represent node of kd tree
struct Node {
    std::vector<float> point;
    int id;
    Node *left;
    Node *right;

    Node(std::vector<float> arr, int setId) : point(arr), id(setId), left(NULL), right(NULL) {}

    ~Node() {
        delete left;
        delete right;
    }
};

struct KdTree {
    Node *root;

    KdTree() : root(NULL) {}

    ~KdTree() { delete root; }

    void insert(std::vector<float> point, int id) {
        // TODO: Fill in this function to insert a new point into the tree
        // the function should create a new node and place correctly with in the root

        // resursive function으로도 만들 수 있지만 그냥 loop 썻다.
        // even depth -> x split, odd depth -> y split
        Node **curNode = &root;
        for (int depth = 0; *curNode; ++depth) {
            if (point.at(depth % 2) < (*curNode)->point.at(depth % 2)) {
                curNode = &((*curNode)->left);
            } else {
                curNode = &((*curNode)->right);
            }
        }
        *curNode = new Node(point, id);
    }

    void searchHelper(
        std::vector<float> target,
        Node *node,
        float distanceTol,
        int depth,
        std::vector<int> &ids) {

        if (!node) {
            return;
        }

        float x_diff = target.at(0) - node->point.at(0);
        float y_diff = target.at(1) - node->point.at(1);

        if (abs(x_diff) < distanceTol && abs(y_diff) < distanceTol) { // 대충 검사
            float distance = sqrt(pow(x_diff, 2) + pow(y_diff, 2));
            if (distance < distanceTol) { // 정밀 검사
                ids.push_back(node->id);
            }
        }

        if ((target.at(depth % 2) - distanceTol) < node->point.at(depth % 2)) {
            searchHelper(target, node->left, distanceTol, depth + 1, ids);
        }
        if ((target.at(depth % 2) + distanceTol) > node->point.at(depth % 2)) {
            searchHelper(target, node->right, distanceTol, depth + 1, ids);
        }
    }

    // TODO: return a list of point ids in the tree that are within distance of target
    std::vector<int> search(std::vector<float> target, float distanceTol) {
        std::vector<int> ids;
        searchHelper(target, root, distanceTol, 0, ids);

        return ids;
    }
};
