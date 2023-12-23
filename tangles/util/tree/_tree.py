# type: ignore
from abc import ABC, abstractmethod

# TODO: Rename file to _graph.py as soon as some branches are merged


class GraphNode(ABC):

    @property
    @abstractmethod
    def neighbours(self):
        pass

    def degree(self):
        return len(self.neighbours)

    def all_nodes(self):
        return list(self.node_iter(depth_first=False))

    def all_edges(self):
        return list(self.edge_iter())

    def edge_iter(self):
        known = set()
        next_nodes = [self]
        while next_nodes:
            node = next_nodes.pop()
            known.add(node)
            for neighbour in node.neighbours:
                if neighbour not in known:
                    yield (node, neighbour)
            next_nodes.extend(set(node.neighbours) - known)

    def node_iter(self, blocked_nodes = None, depth_first=True, predecessors_dict=None):
        pop_idx = -1 if depth_first else 0
        known = set() if blocked_nodes is None else set(blocked_nodes)
        next_nodes = [self]
        known.add(self)
        while next_nodes:
            node = next_nodes.pop(pop_idx)
            unknown_neighbors = [n for n in node.neighbours if n not in known]
            if predecessors_dict is not None:
                predecessors_dict.update(zip(unknown_neighbors,[node]*len(unknown_neighbors)))
            next_nodes.extend(unknown_neighbors)
            known.update(unknown_neighbors)
            yield node

    def shortest_path(self, v:"GraphNode"):
        pred = dict()
        for n in self.node_iter(depth_first=False, predecessors_dict=pred):
            if n==v: break
        if v in pred:
            path = [v]
            while (p:=pred.get(path[-1])):
                path.append(p)
            path.reverse()
            return path
        else:
            return None


    def distance_classes(self):
        prev_shell, shell = set(), {self}
        while shell:
            yield shell
            prev_shell, shell = shell, set().union(*[n.neighbours for n in shell]) - prev_shell

    def search_max_degree_node(self):
        nodes = self.all_nodes()
        degrees = [n.degree() for n in nodes]
        return nodes[degrees.index(max(degrees))]



class TreeNode(GraphNode):
    def is_leaf(self):
        return len(self.neighbours) <= 1

    def all_leaves(self):
        return [n for n in self.node_iter() if n.is_leaf()]

    def split_tree(self):
        return [set(c.node_iter(blocked_nodes={self})) for c in self.neighbours]


