from __future__ import annotations
from colorsys import hsv_to_rgb
from typing import Dict, Iterator, List, Tuple, cast
from random import random
from copy import copy

import pygame
from pygame.math import Vector2

class Unreachable(Exception):
    pass

class Unimplemented(Exception):
    pass

class MalformattedTerm(Exception):
    pass

pygame.init()

minOffset = 1
font = pygame.font.SysFont('verdana', 20)
background = "black"
foreground = "white"
colWidth = 30
rowHeight = 50
startX = 640
startY = 50

def randomColor() -> pygame.Color:
    return pygame.Color(tuple(int(x * 255) for x in hsv_to_rgb(random(), 1, 1)))

def squareRight(v: Vector2) -> Vector2:
    return Vector2(-v.y, v.x)

def surroundingRect(a: Vector2, b: Vector2) -> List[Vector2]:
    basis = (b - a).normalize() * 13
    basis += squareRight(basis) # basis now points left
    return [
        a + squareRight(basis),
        a - basis,
        b - squareRight(basis),
        b + basis
    ]

class Path(str): # if a.path < b.path, then node a is left of node b
    def left(self) -> Path:
        return Path(self + "0")
    
    def down(self) -> Path:
        return Path(self + "1")
    
    def right(self) -> Path:
        return Path(self + "2")
    
    # Cuts off the path after the last "0"
    def lowestLeft(self) -> Path:
        return Path(self[:self.rindex("0") + 1])
    
    # Cuts off the path after the last "2"
    def lowestRight(self) -> Path:
        return Path(self[:self.rindex("2") + 1])

class Node:
    row: int
    xOffset: int
    def __init__(self):
        self.row = 0
        self.xOffset = 0

    def updatePos(self, occupied: NodePositions, row: int = 0, parentX: int = 0, xOffset: int = 0, path: Path = Path()):
        self.row = row
        self.xOffset = xOffset
        
        x = parentX + xOffset

        occupied.occupy(row, x, path)

        match self:
            case Lambda():
                if self.body: self.body.updatePos(occupied, row + 1, x, 0, path.down())
            case Apply():
                if self.left: self.left.updatePos(occupied, row + 1, x, -minOffset, path.left())
                if self.right: self.right.updatePos(occupied, row + 1, x, minOffset, path.right())
                self.highlightRedexes()
            case _:
                pass

    def moveX(self, offset: int):
        self.xOffset += offset

    def updateOccupied(self, occupied: NodePositions, parentX: int = 0, path: Path = Path()):
        pass
    
    def toScreenPos(self, parentX: int) -> Vector2:
        x = parentX + self.xOffset
        return Vector2(x * colWidth + startX, self.row * rowHeight + startY)
    
    def draw(self, screen: pygame.Surface, parentX: int):
        pass

    def undo(self):
        pass

    def isLeaf(self) -> bool:
        return True

    def add(self, new: Node | VarName) -> bool | None:
        return None
    
    def safeDelete(self):
        pass

    def isComplete(self) -> bool:
        return False
    
    # All the following functions are only allowed if the tree is complete

    def copy(self, bindings: VarNameSet) -> Node:
        raise Unreachable()
    
    # Gets the free vars relative to this node
    def getRelFreeVars(self) -> VarNameSet:
        return VarNameSet()

    # only returns true if this node is the var to be sub'd
    def substitute(self, var: VarName, new: Node, relFreeVars: VarNameSet, scope: VarNameMultiSet) -> bool:
        return False
    
    def getLeftmostOutermost(self, path: Path) -> Path | None:
        return None
    
class Lambda(Node):
    param: VarName | None
    body: Node | None
    def __init__(self, param: VarName | None, body: Node | None):
        super().__init__()
        self.param = param
        self.body = body

    def __repr__(self):
        if isinstance(self.body, Lambda):
            return f"λ{self.param}{self.body.__repr__()[1:]}"
        return f"λ{self.param}.{self.body}"

    def updateOccupied(self, occupied: NodePositions, parentX: int = 0, path: Path = Path()):
        x = parentX + self.xOffset
        occupied.occupy(self.row, x, path)
        if self.body: self.body.updateOccupied(occupied, x, path.down())

    def draw(self, screen: pygame.Surface, parentX: int):
        start = self.toScreenPos(parentX)

        if self.body:
            x = parentX + self.xOffset
            end = self.body.toScreenPos(x)
            pygame.draw.line(screen, foreground, start, end)
            self.body.draw(screen, x)
        
        text = font.render(f"λ{self.param if self.param else ''}", True, self.param.color if self.param else foreground, background)
        screen.blit(text, start - Vector2(text.get_size()) / 2)
    
    def undo(self):
        if self.body is None:
            self.param = None
            return
        
        if self.body.isLeaf():
            self.body.safeDelete()
            self.body = None
            return
    
        self.body.undo()

    def isLeaf(self) -> bool:
        return self.param is None
    
    def copy(self, bindings: VarNameSet) -> Node:
        if self.param is None: raise Unreachable()
        newParam = self.param.copy()
        bindings = bindings.addPure(newParam)
        return Lambda(
            newParam,
            self.body.copy(bindings) if self.body else None
        )

    # returns None if nothing was added, True if free var was added, False if otherwise
    def add(self, new: Node | VarName) -> bool | None:
        if self.param is None:
            if type(new) != VarName: return None
            self.param = VarName(new.inner)
            return False
        
        newFree = False
        if type(new) == VarName:
            if new.nameEq(self.param):
                new = self.param
            else:
                newFree = True
        if self.body is None:
            if isinstance(new, VarName): new = Var(new)
            self.body = new
            return newFree
        result = self.body.add(new)
        return result and newFree
    
    def isComplete(self) -> bool:
        return self.body is not None and self.body.isComplete()

    def getRelFreeVars(self) -> VarNameSet:
        if self.body is None or self.param is None: raise Unreachable()
        result = self.body.getRelFreeVars()
        result.remove(self.param)
        return result

    def substitute(self, var: VarName, new: Node, relFreeVars: VarNameSet, scope: VarNameMultiSet) -> bool:
        if self.param is None or self.body is None: raise Unreachable()

        scope.add(self.param)

        if self.body.substitute(var, new, relFreeVars, scope):
            self.body = new.copy(relFreeVars)
        return False
    
    def getLeftmostOutermost(self, path: Path) -> Path | None:
        return cast(Node, self.body).getLeftmostOutermost(path.down())

class Apply(Node):
    left: Node | None
    right: Node | None
    isRedex: bool

    def __init__(self, left: Node | None, right: Node | None):
        super().__init__()
        self.left = left
        self.right = right
        self.isRedex = False

    def __repr__(self):
        left = self.left
        right = self.right
        lhs = f"({left})" if isinstance(left, Lambda) else f"{left}"
        rhs = f"{right}" if isinstance(right, Var) else f"({right})"

        return lhs + " " + rhs

    def updateOccupied(self, occupied: NodePositions, parentX: int = 0, path: Path = Path()):
        x = parentX + self.xOffset
        occupied.occupy(self.row, x, path)
        if self.left: self.left.updateOccupied(occupied, x, path.left())
        if self.right: self.right.updateOccupied(occupied, x, path.right())

    def draw(self, screen: pygame.Surface, parentX: int):
        x = parentX + self.xOffset
        start = self.toScreenPos(parentX)
        
        for node in [self.left, self.right]:
            if node is None: continue
            end = node.toScreenPos(x)
            pygame.draw.line(screen, foreground, start, end)
            node.draw(screen, x)
            
        text = font.render('@', True, foreground, background)
        screen.blit(text, start - Vector2(text.get_size()) / 2)

        if self.isRedex:
            end = cast(Node, self.left).toScreenPos(x)
            pygame.draw.lines(screen, "red", True, surroundingRect(start, end))

    def undo(self):
        if self.right is None:
            left = cast(Node, self.left)
            if left.isLeaf():
                left.safeDelete()
                self.left = None
                return
            
            left.undo()
            return

        if self.right.isLeaf():
            self.right.safeDelete()
            self.right = None
            return

        self.right.undo()

    def isLeaf(self) -> bool:
        return self.left is None
    
    def copy(self, bindings: VarNameSet) -> Node:
        return Apply(
            self.left.copy(bindings) if self.left else None,
            self.right.copy(bindings) if self.right else None
        )

    # returns None if nothing was added, True if free var was added, False if otherwise
    def add(self, new: Node | VarName) -> bool | None:
        if self.left is None:
            if isinstance(new, VarName):
                self.left = Var(new)
                return True
            self.left = new
            return False
        result = self.left.add(new)
        if result is not None:
            return result
        if self.right is None:
            if isinstance(new, VarName):
                self.right = Var(new)
                return True
            self.right = new
            return False
        return self.right.add(new)
    
    def isComplete(self) -> bool:
        return self.right is not None and self.right.isComplete()
    
    def getRelFreeVars(self) -> VarNameSet:
        if self.left is None or self.right is None: raise Unreachable()
        result = self.left.getRelFreeVars()
        result.merge(self.right.getRelFreeVars())
        return result
    
    def highlightRedexes(self):
        self.isRedex = self.left is not None and self.right is not None and isinstance(self.left, Lambda)

    def substitute(self, var: VarName, new: Node, relFreeVars: VarNameSet, scope: VarNameMultiSet) -> bool:
        if self.left is None or self.right is None: raise Unreachable()

        if self.left.substitute(var, new, relFreeVars, scope):
            self.left = new.copy(relFreeVars)
        if self.right.substitute(var, new, relFreeVars, scope):
            self.right = new.copy(relFreeVars)
        
        return False
    
    def getLeftmostOutermost(self, path: Path) -> Path | None:
        if self.isRedex: return path
        return cast(Node, self.left).getLeftmostOutermost(path.left()) or cast(Node, self.right).getLeftmostOutermost(path.right())

class VarName:
    inner: str
    color: pygame.Color
    usage: int
    def __init__(self, name: str):
        self.inner = name
        self.color = randomColor()
        self.usage = 0
    
    def rename(self, newName: str):
        self.inner = newName
    
    def autoRename(self):
        self.inner += "'"

    def incUsage(self):
        self.usage += 1

    def decUsage(self):
        self.usage -= 1

    def copy(self) -> VarName:
        return VarName(self.inner)
    
    def nameEq(self, other: VarName) -> bool:
        return self.inner == other.inner
    
    def __repr__(self):
        return self.inner

class VarNameSet:
    inner: List[VarName]
    def __init__(self, inner: List[VarName] = []):
        self.inner = inner

    def __repr__(self):
        return self.inner.__repr__()

    def add(self, new: VarName):
        matches = [i for i, name in enumerate(self.inner) if name.inner == new.inner]
        if matches:
            self.inner[matches[0]] = new
        else:
            self.inner.append(new)
    
    def addPure(self, new: VarName) -> VarNameSet:
        result = VarNameSet(self.inner.copy())
        result.add(new)
        return result
    
    def remove(self, name: VarName):
        if name in self.inner:
            self.inner.remove(name)

    def merge(self, other: VarNameSet):
        for new in other.inner:
            self.add(new)

    def isEmpty(self) -> bool:
        return len(self.inner) == 0
    
    def clear(self):
        self.inner.clear()

    def removeUnused(self):
        self.inner[:] = [name for name in self.inner if name.usage > 0]

    def get(self, nameStr: str) -> VarName | None:
        result = [v for v in self.inner if v.inner == nameStr]
        if len(result) == 0: return None
        return result[0]

class VarNameMultiSet:
    inner: List[VarName] = []

    def __init__(self):
        self.inner = []
    
    def add(self, new: VarName):
        self.inner.append(new)

    def resolveCollisions(self, others: VarNameSet):
        toRename = self.inner.copy()
        noRename = others.inner.copy()
        toRenameNew: List[VarName] = []
        noRenameNew: List[VarName] = []
        while True:
            for name in toRename:
                for other in noRename:
                    if name.nameEq(other) and name != other:
                        toRenameNew.append(name)
                    else:
                        noRenameNew.append(name)
            if len(toRename) == 0:
                break
            noRename.extend(noRenameNew)
            noRenameNew.clear()
            toRename = toRenameNew
            toRenameNew = []
            for name in toRename:
                name.autoRename()
            toRename.clear()

class Var(Node):
    name: VarName
    def __init__(self, name: VarName):
        super().__init__()
        self.name = name
        name.incUsage()

    def __repr__(self):
        return self.name.__repr__()

    def updateOccupied(self, occupied: NodePositions, parentX: int = 0, path: Path = Path()):
        x = parentX + self.xOffset
        occupied.occupy(self.row, x, path)

    def draw(self, screen: pygame.Surface, parentX: int):
        start = self.toScreenPos(parentX)
        text = font.render(self.name.__repr__(), True, self.name.color, background)
        screen.blit(text, start - Vector2(text.get_size()) / 2)

    def copy(self, bindings: VarNameSet):
        binding = bindings.get(self.name.inner)
        if binding is None: raise Unreachable
        return Var(binding)
    
    def safeDelete(self):
        self.name.decUsage()

    def isComplete(self) -> bool:
        return True
    
    def getRelFreeVars(self) -> VarNameSet:
        return VarNameSet([self.name])

    def substitute(self, var: VarName, new: Node, relFreeVars: VarNameSet, scope: VarNameMultiSet) -> bool:
        if self.name != var:
            return False
        
        scope.resolveCollisions(relFreeVars)
        
        return True

class NodePositions:
    inner: Dict[Tuple[int, int], List[Path]]
    def __init__(self):
        self.inner = {}

    def __repr__(self):
        return self.inner.__repr__()

    def clear(self):
        self.inner.clear()

    def occupy(self, row: int, x: int, path: Path):
        if (row, x) in self.inner:
            self.inner[(row, x)].append(path)
        else:
            self.inner[(row, x)] = [path]

    def getHighestOverlap(self) -> Tuple[Path, Path] | None:
        highest: Tuple[int, List[Path]] | None = None
        for ((row, _), paths) in self.inner.items():
            if len(paths) > 1 and (highest is None or row < highest[0]):
                highest = (row, paths)
        
        if highest is None:
            return None

        paths = highest[1]
        leftMost = min(paths)
        rightMost = max(paths)
        return (leftMost, rightMost)
    
    def getByPos(self, pos: Tuple[int, int]) -> Path | None:
        posLocal = (pos[1] - startY + rowHeight // 2) // rowHeight, (pos[0] - startX + colWidth // 2) // colWidth
        result = [v for k, v in self.inner.items() if k == posLocal]
        return result[0][0] if result and result[0] else None

class Tree:
    root: Node | None
    freeVars: VarNameSet
    nodePositions: NodePositions

    def __init__(self, root: Node | None = None):
        self.root = root
        self.freeVars = VarNameSet()
        self.nodePositions = NodePositions()

    def __repr__(self):
        return self.root.__repr__()

    # Does NOT check if path is valid
    def getByPath(self, path: Path) -> Node:
        currentNode = self.root
        for step in path:
            match step:
                case '0':
                    currentNode = cast(Apply, currentNode).left
                case '1':
                    currentNode = cast(Lambda, currentNode).body
                case '2':
                    currentNode = cast(Apply, currentNode).right
                case _:
                    raise Unreachable()
        return currentNode # type: ignore

    def updateStructure(self):
        self.nodePositions.clear()
        if self.root is None: return
        self.root.updatePos(self.nodePositions)

        i = 0
        while overlap := self.nodePositions.getHighestOverlap():
            self.getByPath(overlap[0].lowestLeft()).moveX(-1)
            self.getByPath(overlap[1].lowestRight()).moveX(1)
            self.nodePositions.clear()
            self.root.updateOccupied(self.nodePositions)
            if i > 2: return
            i += 1
    
    def getByPos(self, pos: Tuple[int, int]) -> Path | None:
        return self.nodePositions.getByPos(pos)

    def betaReduce(self, path: Path):
        if not self.isComplete(): return

        print(self, end=" -> ")

        nodeApp = self.getByPath(path)
        if not (isinstance(nodeApp, Apply) and nodeApp.isRedex): return
        nodeLam = cast(Lambda, nodeApp.left)
        varName = cast(VarName, nodeLam.param)
        subVal = cast(Node, nodeApp.right)
        newRoot = cast(Node, nodeLam.body)
        relFreeVars = subVal.getRelFreeVars()

        if len(path) == 0:
            self.root = newRoot
            parent = self
            scope = VarNameMultiSet()
        else:
            parentPath = Path(path[:-1])
            parent = self.getByPath(parentPath)
            match path[-1]:
                case "0":
                    cast(Apply, parent).left = newRoot
                case "1":
                    cast(Lambda, parent).body = newRoot
                case "2":
                    cast(Apply, parent).right = newRoot
                case _:
                    raise Unreachable()
            scope = self.getScopeByPath(parentPath)

        if newRoot.substitute(varName, subVal, relFreeVars, scope):
            tmp = subVal.copy(relFreeVars)
            if isinstance(parent, Tree):
                self.root = tmp
            else:
                match path[-1]:
                    case "0":
                        cast(Apply, parent).left = tmp
                    case "1":
                        cast(Lambda, parent).body = tmp
                    case "2":
                        cast(Apply, parent).right = tmp
                    case _:
                        raise Unreachable()
        
        subVal.safeDelete()

        self.updateStructure()
        self.freeVars.removeUnused()
        print(self)

    def getScopeByPath(self, path: Path) -> VarNameMultiSet:
        result = VarNameMultiSet()
        current = self.root
        for step in path:
            match step:
                case "0":
                    current = cast(Apply, current).left
                case "1":
                    current = cast(Lambda, current)
                    result.add(cast(VarName, current.param))
                    current = current.body
                case "2":
                    current = cast(Apply, current).right
                case _:
                    raise Unreachable()
        return result

    def autoReduce(self):
        if self.root is None or not self.isComplete(): raise Unreachable()
        path = self.root.getLeftmostOutermost(Path(""))
        if path is not None: self.betaReduce(path)

    def add(self, new: Node | str | VarName):
        if isinstance(new, str):
            new = self.freeVars.get(new) or VarName(new)
            
        if self.root is None:
            if isinstance(new, VarName):
                self.freeVars.add(new)
                new = Var(new)
            self.root = new
            return

        if self.root.add(new):
            self.freeVars.add(cast(VarName, new))

    def addIR(self, ir: str):
        for i, c in enumerate(ir):
            match c:
                case "λ":
                    self.add(Lambda(None, None))
                case "@":
                    self.add(Apply(None, None))
                case "'":
                    continue
                case _:
                    if c in shorthands:
                        self.add(shorthands[c].copy(VarNameSet()))
                        continue
                    j = i + 1
                    while len(ir) > j and ir[j] == "'":
                        c += "'"
                        j += 1
                    self.add(c)

    def draw(self, screen: pygame.Surface):
        if self.root is None: return
        self.root.draw(screen, 0)
    
    def undo(self):
        if self.root:
            if self.root.isLeaf():
                self.root.safeDelete()
                self.root = None
            else:
                self.root.undo()
            self.freeVars.removeUnused()

    def clear(self):
        print("CLEAR")
        self.root = None
        self.freeVars.clear()

    def isComplete(self) -> bool:
        return self.root is not None and self.root.isComplete()
    
    def isClosed(self) -> bool:
        return self.freeVars.isEmpty()

    def copy(self) -> Node | None:
        if self.root is None or not self.isComplete() or not self.isClosed(): return None
        result = self.root.copy(VarNameSet())
        return result

def genChurchNumber(n: int) -> Node:
    z = VarName("z")
    s = VarName("s")
    body = Var(z)
    for _ in range(n):
        body = Apply(Var(s), body)

    return Lambda(
        s,
        Lambda(
            z,
            body
        )
    )

def tryNext(it: Iterator[str]) -> str:
    try:
        return next(it)
    except StopIteration:
        return ""

def peek(it: Iterator[str]) -> str:
    return tryNext(copy(it))

# @returns the subtree and any free vars in the sub term
def _parseTerm(term: Iterator[str]) -> str:
    subTerms = ""
    nApply = 0
    while True:
        match c := tryNext(term):
            case "(":
                subTerms += _parseTerm(term)
            case "L" | "λ":
                subTerms += _parseLambda(term)
                return "@" * nApply + "".join(subTerms)
            case _:
                if (c < "a" or c > "z") and (c < "A" or c > "Z"): raise MalformattedTerm(c)
                while peek(term) == "'":
                    c += next(term)
                subTerms += c
                
        match c := tryNext(term):
            case "" | ")":
                return "@" * nApply + "".join(subTerms)
            case " ":
                nApply += 1
            case _:
                raise MalformattedTerm(c)

# Starts after the 'L'        
def _parseLambda(term: Iterator[str]) -> str:
    params = ""
    while (c := tryNext(term)) != ".":
        if c < "a" or c > "z": raise MalformattedTerm(c)
        while peek(term) == "'":
            c += next(term)
        params += "λ" + c
    
    return params + _parseTerm(term)

shorthands: Dict[str, Node] = {}

def parseTerm(term: str) -> Tree | None:
    try:
        ir = _parseTerm(iter(term))
    except MalformattedTerm:
        return None
    result = Tree()
    result.addIR(ir)

    result.updateStructure()
    return result

with open("shorthands") as fh:
    for line in fh.readlines():
        shorthand, term = line.split(" = ")
        termTree = parseTerm(term.strip())
        if termTree is not None:
            shorthands[shorthand] = cast(Node, termTree.root)

tree: Tree = Tree()

screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
isShiftPressed = False

def getUsage() -> List[str]:
    if isShiftPressed:
        if tree.isComplete():
            return ["Shift+Letter: Save current tree as shorthand"]
        return ["Insert shorthand:"] + [f"{k}: {v}" for k, v in shorthands.items()] + ["0-9: Church numbers"]
    
    result = [
        "Escape: exit",
        "Backspace: remove last node",
        "Delete: delete tree",
    ]

    if tree.isComplete():
        result += [
            "Enter: β-reduce"
        ]
    else:
        result += [
            "L: Lambda node",
            "Space: Apply node",
            "Any letter: Var node",
            "Enter: insert term from terminal"
        ]
    return result
    
def saveShorthand(name: str):
    newShorthand = tree.copy()
    if newShorthand:
        shorthands[name] = newShorthand
        with open("shorthands", "r+") as fh:
            lines = [line for line in fh.readlines() if line[0] != name]
            lines.append(f"{name} = {newShorthand}\n".replace("λ", "L"))
            fh.seek(0)
            fh.writelines(lines)
            fh.truncate()
        print(f"Added shorthand {name} = {newShorthand}")

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window

    newNode: Node | str | None = None

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LSHIFT:
                isShiftPressed = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LSHIFT:
                isShiftPressed = False
                continue
            if event.mod & pygame.KMOD_SHIFT:
                if event.key >= pygame.K_0 and event.key <= pygame.K_9:
                    newNode = genChurchNumber(event.key - pygame.K_0)
                    continue
                if event.unicode < 'A' or event.unicode > 'Z': continue
                if tree.isComplete():
                    saveShorthand(event.unicode)
                    continue
                
                if event.unicode in shorthands:
                    newNode = shorthands[event.unicode].copy(VarNameSet())
                continue
            match event.key:
                case pygame.K_ESCAPE:
                    running = False
                case pygame.K_DELETE:
                    tree.clear()
                case pygame.K_BACKSPACE:
                    tree.undo()
                    tree.updateStructure()
                case pygame.K_RETURN:
                    if tree.isComplete():
                        tree.autoReduce()
                        continue
                    screen.blit(font.render("Awaiting input from terminal", True, "red"), (200,10))
                    pygame.display.flip()
                    inp = input("Term (use L for λ): ").strip()
                    try:
                        ir = _parseTerm(iter(inp))
                    except MalformattedTerm:
                        print("Term was invalid")
                        continue
                    tree.addIR(ir)
                    tree.updateStructure()
                case pygame.K_l:
                    newNode = Lambda(None, None)
                case pygame.K_SPACE:
                    newNode = Apply(None, None)
                case _:
                    if event.unicode != "": newNode = event.unicode
        
        if event.type == pygame.MOUSEBUTTONUP:
            path = tree.getByPos(event.pos)
            if path is not None:
                tree.betaReduce(path)

        if event.type == pygame.VIDEORESIZE:
            startX = screen.get_width() // 2

    if newNode:
        tree.add(newNode)
        tree.updateStructure()

    # fill the screen with a color to wipe away anything from last frame
    screen.fill(background)

    tree.draw(screen)

    usageSurfs = [(font.render(line, True, foreground), (10, 10+25*i)) for i, line in enumerate(getUsage())]
    screen.blits(usageSurfs)

    if tree.isComplete():
        termSurf = font.render(tree.__repr__(), True, foreground)
        screen.blit(termSurf, (startX * 2 - termSurf.get_width() - 10, 10))

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()

