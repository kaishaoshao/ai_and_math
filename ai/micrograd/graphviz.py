from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prew:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    '''
    format: png | svg | ...
    rankdir: TB (top tp bottom graph) | LR (left to right)
    '''
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    

