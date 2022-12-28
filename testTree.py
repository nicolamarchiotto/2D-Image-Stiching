
from anytree import Node, RenderTree, Resolver, PreOrderIter

r = Resolver('name')
head = Node(str(4))

marc = Node("8", h=[1], parent=head)


# marc2 = r.get(head, "1")
# son_of_marc = Node("2", h=[2], parent=marc2)
# son_of_marc = Node("3", h=[3], parent=marc2)
# son_of_marc = Node("4", h=[4], parent=head)


# for pre, fill, node in RenderTree(head):
#     print("%s%s" % (pre, node.name))

# print("chi ",marc.children)
# l = marc.children[0]
# print("l ",l)
# print("l ",l.parent)

for node in PreOrderIter(head):
    print(node.name)

