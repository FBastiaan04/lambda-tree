from __future__ import annotations
from colorsys import hsv_to_rgb
from typing import Dict, List, Tuple, cast
from random import random

import pygame
from pygame.math import Vector2

class Unreachable(Exception):
    pass

class Unimplemented(Exception):
    pass

pygame.init()

minOffset = 1
font = pygame.font.SysFont('arial', 16)

def randomColor() -> pygame.Color:
    return pygame.Color(tuple(int(x * 255) for x in hsv_to_rgb(random(), 1, 1)))

class Path(str): # if a.path < b.path, then node a is left of node b
    def __repr__(self):
        return self.replace("0", "L").replace("1", "D").replace("2", "R")
    
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
            case _:
                raise Unreachable()

    def moveX(self, offset: int):
        self.xOffset += offset

    def updateOccupied(self, occupied: NodePositions, parentX: int = 0, path: Path = Path()):
        pass
    
    def toScreenPos(self, parentX: int) -> Vector2:
        x = parentX + self.xOffset
        return Vector2(x * 30 + 500, self.row * 50 + 100)
    
    def draw(self, screen: pygame.Surface, parentX: int):
        pass

    def undo(self):
        pass

    def isLeaf(self) -> bool:
        return True
    
    def copy(self, bindings: VarNameSet) -> Node:
        raise Unreachable()

    def add(self, new: Node | VarName) -> bool | None:
        return None
    
    def safeDelete(self):
        pass

class Lambda(Node):
    param: VarName | None
    body: Node | None
    def __init__(self, param: VarName | None, body: Node | None):
        super().__init__()
        self.param = param
        self.body = body

    def __repr__(self):
        return f"L{self.param}.{self.body}"

    def updateOccupied(self, occupied: NodePositions, parentX: int = 0, path: Path = Path()):
        x = parentX + self.xOffset
        occupied.occupy(self.row, x, path)
        if self.body: self.body.updateOccupied(occupied, x, path.down())

    def draw(self, screen: pygame.Surface, parentX: int):
        start = self.toScreenPos(parentX)

        if self.body:
            x = parentX + self.xOffset
            end = self.body.toScreenPos(x)
            pygame.draw.line(screen, "black", start, end)
            self.body.draw(screen, x)
        
        text = font.render(f"λ{self.param if self.param else ''}", True, self.param.color if self.param else "black", "purple")
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
            
class Apply(Node):
    left: Node | None
    right: Node | None
    def __init__(self, left: Node | None, right: Node | None):
        super().__init__()
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left}) ({self.right})"

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
            pygame.draw.line(screen, "black", start, end)
            node.draw(screen, x)
            
        text = font.render('@', True, "black", "purple")
        screen.blit(text, start - Vector2(text.get_size()) / 2)

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

class VarName:
    inner: str
    color: pygame.Color
    usage: int
    def __init__(self, name: str):
        self.inner = name
        self.color = randomColor()
        self.usage = 0
    
    def rename(self, newName: str):
        self.name = newName
    
    def autoRename(self):
        self.name += "'"

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

class Var(Node):
    name: VarName
    def __init__(self, name: VarName):
        super().__init__()
        self.name = name
        self.name.incUsage()

    def __repr__(self):
        return self.name.__repr__()

    def updateOccupied(self, occupied: NodePositions, parentX: int = 0, path: Path = Path()):
        x = parentX + self.xOffset
        occupied.occupy(self.row, x, path)

    def draw(self, screen: pygame.Surface, parentX: int):
        start = self.toScreenPos(parentX)
        text = font.render(self.name.__repr__(), True, self.name.color, "purple")
        screen.blit(text, start - Vector2(text.get_size()) / 2)

    def copy(self, bindings: VarNameSet):
        binding = bindings.get(self.name.inner)
        if binding is None: raise Unreachable
        return Var(binding)
    
    def safeDelete(self):
        self.name.decUsage()

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
        self.root = None
        self.freeVars.clear()

    def isComplete(self) -> bool:
        return False
    
    def isClosed(self) -> bool:
        return self.freeVars.isEmpty()

    def copy(self) -> Node | None:
        if self.root is None or not self.isComplete() or not self.isClosed(): return None
        result = self.root.copy(VarNameSet())
        return result

tree = Tree()

screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

shorthandValsX = [VarName("x") for _ in range(3)]
shorthandValsY = [VarName("y") for _ in range(2)]

shorthands: Dict[str, Node] = {
    "I": Lambda(
        shorthandValsX[0],
        Var(shorthandValsX[0])
    ),
    "T": Lambda(
        shorthandValsX[1],
        Lambda(
            shorthandValsY[0],
            Var(shorthandValsX[1])
        )
    ),
    "F": Lambda(
        shorthandValsX[2],
        Lambda(
            shorthandValsY[1],
            Var(shorthandValsY[1])
        )
    )
}

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

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window

    newNode: Node | str | None = None

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYUP:
            print(event)
            if event.mod & pygame.KMOD_SHIFT:
                if event.key < 33 or event.key > 126: continue
                if event.unicode in shorthands:
                    newNode = shorthands[event.unicode].copy(VarNameSet())
                elif event.key >= pygame.K_0 and event.key <= pygame.K_9:
                    newNode = genChurchNumber(event.key - pygame.K_0)
                else:
                    newShorthand = tree.copy()
                    print(f"Added shorthand {newShorthand} as {event.unicode}")
                    if newShorthand: shorthands[event.unicode] = newShorthand
                continue
            match event.key:
                case pygame.K_ESCAPE:
                    running = False
                case pygame.K_DELETE:
                    tree.clear()
                case pygame.K_BACKSPACE:
                    tree.undo()
                    tree.updateStructure()
                case pygame.K_l:
                    newNode = Lambda(None, None)
                case pygame.K_SPACE:
                    newNode = Apply(None, None)
                case _:
                    if event.unicode != "": newNode = event.unicode

    if newNode:
        print(f"adding {newNode}")
        tree.add(newNode)
        tree.updateStructure()
        print(tree, tree.freeVars)

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("purple")

    tree.draw(screen)

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()
