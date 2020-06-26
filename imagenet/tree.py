import xmltodict


class Node(object):
    def __init__(self, node, parent=None):
        self._node = node
        self.wnid = node['@wnid']
        self.gloss = node['@gloss']
        self.name = node['@words']
        self.parent = parent
        self._set_children_node()

    def _set_children_node(self):
        self.children = []
        if 'synset' in self._node:
            children = self._node['synset']
            if not isinstance(children, list):
                children = [children]
            for child in children:
                child = Node(child, parent=self)
                #if child.numImg > 0:
                self.children.append(child)

        self.num_children = len(self.children)

    def is_leaf(self):
        return self.num_children == 0

    def __repr__(self):
        return self.name


class Tree(object):
    def __init__(self, root):
        self.root = root

    def find_node(self, name, node=None):
        node = self.root if node is None else node

        if node and node.name == name:
            return node
        if not node.is_leaf():
            for child in node.children:
                node = self.find_node(name, child)
                if node and node.name == name:
                    return node

    def find_all_children(self, node=None):
        node = self.root if node is None else node
        children = []

        def _dfs(node):
            children.append(node)
            if not node.is_leaf():
                for child in node.children:
                    node = _dfs(child)

        _dfs(node)
        return children

    def find_parents(self, children):

        children_to_parent = {}

        def _dfs(node, parent_node):
            if node in children:
                children_to_parent.update({node: parent_node})

            if not node.is_leaf():
                for child in node.children:
                    _dfs(child, node)

        _dfs(self.root, None)
        return [children_to_parent[c] for c in children]

    def find_max_depth(self, node=None):
        node = self.root if node is None else node
        depth = []

        def _dfs(node, cur_depth):
            cur_depth += 1
            depth.append(cur_depth)

            if not node.is_leaf():
                for child in node.children:
                    _dfs(child, cur_depth)

        _dfs(node, cur_depth=0)
        return max(depth)


def get_imagenet_structure(path):

    with open(path) as f:
        xml = f.read()
        struct = xmltodict.parse(xml)
    root = struct['ImageNetStructure']['synset']

    root_node = Node(root)
    tree = Tree(root_node)
    return tree
