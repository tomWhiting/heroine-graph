# 12
## Force-Directed Drawing Algorithms


Stephen G. Kobourov
University of Arizona


12.1 Introduction ................................................. 383
12.2 Spring Systems and Electrical Forces ................... 385
12.3 The Barycentric Method .................................. 386
12.4 Graph Theoretic Distances Approach ................... 388
12.5 Further Spring Refinements............................... 389
12.6 Large Graphs ............................................... 391
12.7 Stress Majorization ........................................ 396
12.8 Non-Euclidean Approaches ............................... 397
12.9 Lombardi Spring Embedders ............................. 400
12.10 Dynamic Graph Drawing ................................. 401
12.11 Conclusion .................................................. 403
References .......................................................... 404


### 12.1 Introduction

Some of the most flexible algorithms for calculating layouts of simple undirected graphs
belong to a class known as force-directed algorithms. Also known as spring embedders,
such algorithms calculate the layout of a graph using only information contained within
the structure of the graph itself, rather than relying on domain-specific knowledge. Graphs
drawn with these algorithms tend to be aesthetically pleasing, exhibit symmetries, and tend
to produce crossing-free layouts for planar graphs. In this chapter we will assume that the
input graphs are simple, connected, undirected graphs and their layouts are straight-line
drawings. Excellent surveys of this topic can be found in Di Battista et al. [DETT99]
Chapter 10 and Brandes [Bra01].
Going back to 1963, the graph drawing algorithm of Tutte [Tut63] is one of the first forcedirected graph drawing methods based on barycentric representations. More traditionally,
the spring layout method of Eades [Ead84] and the algorithm of Fruchterman and Reingold [FR91] both rely on spring forces, similar to those in Hooke’s law. In these methods,
there are repulsive forces between all nodes, but also attractive forces between nodes that
are adjacent.
Alternatively, forces between the nodes can be computed based on their graph theoretic
distances, determined by the lengths of shortest paths between them. The algorithm of
Kamada and Kawai [KK89] uses spring forces proportional to the graph theoretic distances.
In general, force-directed methods define an objective function which maps each graph
layout into a number in R [+] representing the energy of the layout. This function is defined
in such a way that low energies correspond to layouts in which adjacent nodes are near some
pre-specified distance from each other, and in which non-adjacent nodes are well-spaced. A


383


384 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS


Figure 12.1 Examples of drawings obtained with force-directed algorithms. First row:
small graphs: dodecahedron (20 vertices), C60 bucky ball (60 vertices), 3D cube mesh (216
vertices). Second row: Cubes in 4D, 5D and 6D [GK02].


layout for a graph is then calculated by finding a (often local) minimum of this objective
function; see Figure 12.1.
The utility of the basic force-directed approach is limited to small graphs and results are
poor for graphs with more than a few hundred vertices. There are multiple reasons why
traditional force-directed algorithms do not perform well for large graphs. One of the main
obstacles to the scalability of these approaches is the fact that the physical model typically
has many local minima. Even with the help of sophisticated mechanisms for avoiding local
minima the basic force-directed algorithms are not able to consistently produce good layouts
for large graphs. Barycentric methods also do not perform well for large graphs mainly due
to resolution problems: for large graphs the minimum vertex separation tends to be very
small, leading to unreadable drawings.
The late 1990s saw the emergence of several techniques extending the functionality of
force-directed methods to graphs with tens of thousands and even hundreds of thousands of
vertices. One common thread in these approaches is the multi-level layout technique, where
the graph is represented by a series of progressively simpler structures and laid out in reverse
order: from the simplest to the most complex. These structures can be coarser graphs (as in
the approach of Hadany and Harel [HH01], Harel and Koren [HK02], and Walshaw [Wal03],
or vertex filtrations as in the approach of Gajer, Goodrich, and Kobourov [GGK04].
The classical force-directed algorithms are restricted to calculating a graph layout in
Euclidean geometry, typically R [2], R [3], and, more recently, R [n] for larger values of n. There
are, however, cases where Euclidean geometry may not be the best option: Certain graphs
may be known to have a structure which would be best realized in a different geometry,


12.2. SPRING SYSTEMS AND ELECTRICAL FORCES 385


such as on the surface of a sphere or on a torus. In particular, 3D mesh data can be
parameterized on the sphere for texture mapping or graphs of genus one can be embedded on
a torus without crossings. Furthermore, it has also been noted that certain non- Euclidean
geometries, specifically hyperbolic geometry, have properties that are particularly well suited
to the layout and visualization of large classes of graphs [LRP95, Mun97]. With this in mind,
Kobourov and Wampler describe extensions of the force-directed algorithms to Riemannian
spaces [KW05].

### 12.2 Spring Systems and Electrical Forces


The 1984 algorithm of Eades [Ead84] targets graphs with up to 30 vertices and uses a
mechanical model to produce “aesthetically pleasing” 2D layouts for plotters and CRT
screens. The algorithm is succinctly summarized as follows:


To embed a graph we replace the vertices by steel rings and replace each edge with
a spring to form a mechanical system. The vertices are placed in some initial
layout and let go so that the spring forces on the rings move the system to a
minimal energy state. Two practical adjustments are made to this idea: firstly,
logarithmic strength springs are used; that is, the force exerted by a spring is:


c1 ∗ log(d/c2),


where d is the length of the spring, and c1 and c2 are constants. Experience
shows that Hookes Law (linear) springs are too strong when the vertices are far
apart; the logarithmic force solves this problem. Note that the springs exert no
force when d = c2. Secondly, we make nonadjacent vertices repel each other. An
inverse square law force,
c3/d [2],


where c3 is constant and d is the distance between the vertices, is suitable. The
mechanical system is simulated by the following algorithm.


algorithm SPRING(G:graph);
place vertices of G in random locations;
repeat M times
calculate the force on each vertex;
move the vertex c4 ∗ (force on vertex)
draw graph on CRT or plotter.


The values c1 = 2, c2 = 1, c3 = 1, c4 = 0.1, are appropriate for most graphs.
Almost all graphs achieve a minimal energy state after the simulation step is
run 100 times, that is, M = 100.


This excellent description encapsulates the essence of spring algorithms and their natural
simplicity, elegance, and conceptual intuitiveness. The goals behind “aesthetically pleasing”
layouts were initially captured by the two criteria: “all the edge lengths ought to be the
same, and the layout should display as much symmetry as possible.”
The 1991 algorithm of Fruchterman and Reingold added “even vertex distribution” to the
earlier two criteria and treats vertices in the graph as “atomic particles or celestial bodies,


386 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS


exerting attractive and repulsive forces from one another.” The attractive and repulsive
forces are redefined to


fa(d) = d [2] /k, fr(d) = −k [2] /d,


in terms of the distance d between two vertices and the optimal distance between vertices
k defined as ~~�~~

~~area~~
k = C
number of vertices [.]


This algorithm is similar to that of Eades in that both algorithms compute attractive
forces between adjacent vertices and repulsive forces between all pairs of vertices. The
algorithm of Fruchterman and Reingold adds the notion of “temperature” which could
be used as follows: “the temperature could start at an initial value (say one tenth the
width of the frame) and decay to 0 in an inverse linear fashion.” The temperature controls
the displacement of vertices so that as the layout becomes better, the adjustments become
smaller. The use of temperature here is a special case of a general technique called simulated
annealing, whose use in force-directed algorithms is discussed later in this chapter. The
pseudo-code for the algorithm by Fruchterman and Reingold, shown in Figure 12.2 provides
further insight into the workings of a spring-embedder.
Each iteration the basic algorithm computes O(|E|) attractive forces and O(|V | [2] ) repulsive forces. To reduce the quadratic complexity of the repulsive forces, Fruchterman and
Reingold suggest using a grid variant of their basic algorithm, where the repulsive forces between distant vertices are ignored. For sparse graphs, and with uniform distribution of the
vertices, this method allows a O(|V |) time approximation to the repulsive forces calculation.
This approach can be thought of as a special case of the multi-pole technique introduced in
n-body simulations [Gre88] whose use in force-directed algorithms will be further discussed
later in this chapter.
As in the paper by Eades [Ead84] the graphs considered by Fruchterman and Reingold
are small graphs with less than 40 vertices. The number of iterations of the main loop is
also similar at 50.

### 12.3 The Barycentric Method


Historically, Tutte’s 1963 barycentric method [Tut63] is the first “force-directed” algorithm
for obtaining a straight-line, crossings free drawing for a given 3-connected planar graph.
Unlike almost all other force-directed methods, Tutte’s guarantees that the resulting drawing is crossings-free; moreover, all faces of the drawing are convex.
The idea behind Tutte’s algorithm, shown in Figure 12.3, is that if a face of the planar
graph is fixed in the plane, then suitable positions for the remaining vertices can be found by
solving a system of linear equations, where each vertex position is represented as a convex
combination of the positions of its neighbors. This algorithm be considered a force-directed
method as summarized in Di Battista et al. [DETT99].
In this model the force due to an edge (u, v) is proportional to the distance between
vertices u and v and the springs have ideal length of zero; there are no explicit repulsive
forces. Thus the force at a vertex v is described by

       F (v) = (pu − pv),

(u,v)∈E


where pu and pv are the positions of vertices u and v. As this function has a trivial minimum
with all vertices placed in the same location, the vertex set is partitioned into fixed and free


12.3. THE BARYCENTRIC METHOD 387


area:= W ∗ L; {W and L are the width and length of the frame}
G := ( ~~�~~ V, E); {the vertices are assigned random initial positions}
k := area/|V |;

function fa(x) := begin return x [2] /k end;
function fr(x) := begin return k [2] /x end;
for i := 1 to iterations do begin
{calculate repulsive forces}
for v in V do begin
{each vertex has two vectors: .pos and .disp
v.disp := 0;
for u in V do
if (u ̸= v) then begin
{δ is the difference vector between the positions of the two vertices}
δ := v.pos − u.pos;
v.disp := v.disp + (δ/|δ|) ∗ fr(|δ|)
end
end
{calculate attractive forces}
for e in E do begin
{each edges is an ordered pair of vertices .vand.u}
δ := e.v.pos − e.u.pos;
e.v.disp := e.v.disp − (δ/|δ|) ∗ fa(|δ|);
e.u.disp := e.u.disp + (δ/|δ|) ∗ fa(|δ|)
end
{limit max displacement to temperature t and prevent from displacement
outside frame}
for v in V do begin
v.pos := v.pos + (v.disp/|v.disp|) ∗ min(v.disp, t);
v.pos.x := min(W/2, max(−W/2, v.pos.x));
v.pos.y := min(L/2, max(−L/2, v.pos.y))
end
{reduce the temperature as the layout approaches a better configuration}
t := cool(t)
end


Figure 12.2 Pseudo-code for the algorithm by Fruchterman and Reingold [FR91].


vertices. Setting the partial derivatives of the force function to zero results in independent
systems of linear equations for the x-coordinate and for the y-coordinate.


The equations in the for-loop are linear and the number of equations is equal to the
number of the unknowns, which in turn is equal to the number of free vertices. Solving these
equations results in placing each free vertex at the barycenter of its neighbors. The system
of equations can be solved using the Newton-Raphson method. Moreover, the resulting
solution is unique.


One significant drawback of this approach is the resulting drawing often has poor vertex
resolution. In fact, for every n > 1, there exists a graph, such that the barycenter method
computes a drawing with exponential area [EG95].


388 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS


Barycenter-Draw
Input: G = (V, E); a partition V = V0 ∪ V1 of V into a set V0 of at least three
fixed vertices and a set V1 of free vertices; a strictly convex polygon P with |V0|
vertices
Output: a position pv for each vertex of V, such that the fixed vertices form a
convex polygon P .


1. Place each fixed vertex u ∈ V0 at a vertex of P, and each free vertex at the
origin.


2. repeat
foreach free vertex v ∈ V1 do


1
xv =
deg(v)


1
yv =
deg(v)



xu

(u,v)∈E



yu

(u,v)∈E


until xv and yv converge for all free vertices v.


Figure 12.3 Tutte’s barycentric method [Tut63]. Pseudo-code from [DETT99].

### 12.4 Graph Theoretic Distances Approach


The 1989 algorithm of Kamada and Kawai [KK89] introduced a different way of thinking
about “good” graph layouts. Whereas the algorithms of Eades and Fruchterman-Reingold
aim to keep adjacent vertices close to each other while ensuring that vertices are not too
close to each other, Kamada and Kawai take graph theoretic approach:


We regard the desirable geometric (Euclidean) distance between two vertices in
the drawing as the graph theoretic distance between them in the corresponding
graph.


In this model, the “perfect” drawing of a graph would be one in which the pair-wise geometric distances between the drawn vertices match the graph theoretic pairwise distances,
as computed by an All-Pairs-Shortest-Path computation. As this goal cannot always be
achieved for arbitrary graphs in 2D or 3D Euclidean spaces, the approach relies on setting
up a spring system in such a way that minimizing the energy of the system corresponds to
minimizing the difference between the geometric and graph distances. In this model there
are no separate attractive and repulsive forces between pairs of vertices, but instead if a
pair of vertices is (geometrically) closer/farther than their corresponding graph distance the
vertices repel/attract each other. Let di,j denote the shortest path distance between vertex
i and vertex j in the graph. Then li,j = L × di,j is the ideal length of a spring between
vertices i and j, where L is the desirable length of a single edge in the display. Kamada


12.5. FURTHER SPRING REFINEMENTS 389


and Kawai suggest that L = L0/ maxi<j di,j, where L0 is the length of a side of the display
area and maxi<j di,j is the diameter of the graph, i.e., the distance between the farthest
pair of vertices. The strength of the spring between vertices i and j is defined as


ki,j = K/d [2] i,j [,]


where K is a constant. Treating the drawing problem as localizing |V | = n particles
p1, p2, . . ., pn in 2D Euclidean space, leads to the following overall energy function:


E =


n�−1 �n


i=1 j=i+1


1
2 [k][i,j][(][|][p][i][ −] [p][j][| −] [l][i,j][)][2][.]


The coordinates of a particle pi in the 2D Euclidean plane are given by xi and yi which
allows us to rewrite the energy function as follows:


- ~~�~~ (xi − xj) [2] + (yi − yj) [2] + li,j [2] [−] [2][l][i,j] (xi − xj) [2] + (yi − yj) [2] .


E =


n�−1


i=1


�n


j=i+1


1
2 [k][i,j]


The goal of the algorithm is to find values for the variables that minimize the energy
function E(x1, x2, . . ., xn, y1, y2, . . ., yn). In particular, at a local minimum all the partial
derivatives are equal to zero, and which corresponds to solving 2n simultaneous non-linear
equations. Therefore, Kamada and Kawai compute a stable position one particle pm at
a time. Viewing E as a function of only xm and ym a minimum of E can be computed
using the Newton-Raphson method. At each step of the algorithm the particle pm with the
largest value of ∆m is chosen, where


~~�~~ 2 ~~�~~
∂E
+

∂ym


~~�~~ 2
.


∆m =


~~��~~
∂E

∂xm


Pseudo-code for the algorithm by Kamada and Kawai is shown in Figure 12.4.
The algorithm of Kamada and Kawai is computationally expensive, requiring an All-PairShortest-Path computation which can be done in O(|V | [3] )time using the Floyd-Warshall algorithm or in O(|V | [2] log |V | + |E||V |) using Johnson’s algorithm; see the All-Pairs-ShortestPath chapter in an algorithms textbook such as [CLRS90]. Furthermore, the algorithm
requires O(|V | [2] ) storage for the pairwise vertex distances. Despite the higher time and
space complexity, the algorithm contributes a simple and intuitive definition of a “good”
graph layout: A graph layout is good if the geometric distances between vertices closely
correspond to the underlying graph distances.

### 12.5 Further Spring Refnements


Even before the 1984 algorithm of Eades, force-directed techniques were used in the context
of VLSI layouts in the 1960s and 1970s [FCW67, QB79]. Yet, renewed interest in forcedirected graph layout algorithms brought forth many new ideas in the 1990s. Frick, Ludwig,
and Mehldau [FLM95] add new heuristics to the Fruchterman-Reingold approach. In particular, oscillation and rotations are detected and dealt with using local instead of global
temperature. The following year Bruß and Frick [BF96] extended the approach to layouts
directly in 3D Euclidean space. The algorithm of Cohen [Coh97] introduced the notion of
an incremental layout, a precursor of the multi-scale methods described in Section 12.6.


390 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS


compute di,j for 1 ≤ i ̸= j ≤ n;
compute li,j for 1 ≤ i ̸= j ≤ n;
compute ki,j for 1 ≤ i ̸= j ≤ n;
initialize p1, p2, . . ., pn;
while (maxi∆i > ǫ)
let pm be the particle satisfying ∆m = maxi∆i;
while (∆m > ǫ)
compute δx and δy by solving the following system of equations:


∂ [2] E ∂ [2] E
(x [(] m [t][)][, y] m [(][t][)][)][δx][ +] (x [(] m [t][)][, y] m [(][t][)][)][δy][ =][ −] [∂E] (x [(] m [t][)][, y] m [(][t][)][);]
∂x [2] m ∂xm∂ym ∂xm


∂ [2] E
(x [(] m [t][)][, y] m [(][t][)][)][δx][ +] [∂][2][E]
∂ym∂xm ∂ym [2]


(x [(] m [t][)][, y] m [(][t][)][)]
∂ym


[∂][2][E] (x [(] m [t][)][, y] m [(][t][)][)][δy][ =][ −] [∂E]

∂ym [2] ∂y


xm := xm + δx;
ym := ym + δy;


Figure 12.4 Pseudo-code for the algorithm by Kamada and Kawai [KK89].


The 1997 algorithm of Davidson and Harel [DH96] adds additional constraints to the
traditional force-directed approach in explicitly aiming to minimize the number of edgecrossings and keeping vertices from getting too close to non-adjacent edges. The algorithm uses the simulated annealing technique developed for large combinatorial optimization [KGV83]. Simulated annealing is motivated by the physical process of cooling molten
materials. When molten steel is cooled too quickly it cracks and forms bubbles making it
brittle. For better results, the steel must be cooled slowly and evenly and this process is
known as annealing in metallurgy. With regard to force-directed algorithms, this process is
simulated to find local minima of the energy function. Cruz and Twarog [CT96] extended
the method by Davidson and Harel to three-dimensional drawings.
Genetic algorithms for force-directed placement have also been considered. Genetic algorithms are a commonly used search technique for finding approximate solutions to optimization and search problems. The technique is inspired by evolutionary biology in general
and by inheritance, mutation, natural selection, and recombination (or crossover), in particular; see the survey by Vose [Vos99]. In the context of force-directed techniques for
graph drawing, the genetic algorithms approach was introduced in 1991 by Kosak, Marks
and Shieber [KMS91]. Other notable approaches in the direction include that of Branke,
Bucher, and Schmeck [BBS97].
In the context of graph clustering, the LinLog model introduces an alternative energy
model [Noa07]. Traditional energy models enforce small and uniform edge lengths, which
often prevent the separation of nodes in different clusters. As a side effect, they tend
to group nodes with large degree in the center of the layout, where their distance to the
remaining nodes is relatively small. The node-repulsion LinLog and edge- repulsion LinLog
models group nodes according to two well-known clustering criteria: the density of the
cut [LR88] and the normalized cut [SM00].


12.6. LARGE GRAPHS 391

### 12.6 Large Graphs


The first force-directed algorithms to produce good layouts for graphs with more than 1000
vertices is the 1999 algorithm of Hadany and Harel [HH01]. They introduced the multiscale technique as a way to deal with large graphs and in the following year four related
but independent force-directed algorithms for large graphs were presented at the Annual
Symposium on Graph Drawing. We begin with Hadany and Harel’s description on the
multi-scale method :


A natural strategy for drawing a graph nicely is to first consider an abstraction,
disregarding some of the graph’s fine details. This abstraction is then drawn,
yielding a “rough” layout in which only the general structure is revealed. Then
the details are added and the layout is corrected. To employ such a strategy
it is crucial that the abstraction retains essential features of the graph. Thus,
one has to define the notion of coarse-scale representations of a graph, in which
the combinatorial structure is significantly simplified but features important for
visualization are well preserved. The drawing process will then “travel” between
these representations, and introduce multi-scale corrections. Assuming we have
already defined the multiple levels of coarsening, the general structure of our
strategy is as follows:


1. Perform fine-scale relocations of vertices that yield a locally organized configuration.


2. Perform coarse-scale relocations (through local relocations in the coarse representations), correcting global disorders not found in stage 1.


3. Perform fine-scale relocations that correct local disorders introduced by stage 2.


Hadany and Harel suggest computing the sequence of graphs by using edge contractions
so as to preserve certain properties of the graph. In particular, the goal is to preserve three
topological properties: cluster size, vertex degrees, and homotopy. For the coarse-scale
relocations, the energy function for each graph in the sequence is that of Kamada and Kawai
(the pairwise graph distances are compared to the geometric distances in the current layout).
For the fine-scale relocations, the authors suggest using force-directed calculations as those
of Eades [Ead84], Fruchterman-Reingold [FR91], or Kamada-Kawai [KK89]. While the
asymptotic complexity of this algorithm is similar to that of the Kamada-Kawai algorithm,
the multi-scale approach leads to good layouts for much larger graphs in reasonable time.
The algorithm of Harel and Koren [HK02] took force-directed algorithms to graphs with
15,000 vertices. This algorithm is similar to the algorithm of Hadany and Harel, yet uses
a simpler coarsening process based on a k-centers approximation, and a faster fine-scale
beautification. Given a graph G = (V, E), the k-centers problem asks to find a subset of
the vertex set V [′] ⊆ V of size k, so as to minimize the maximum distance from a vertex to
V [′] : min u ∈ V maxu∈V,v∈V ′ dist(u, v). While k-centers is an NP-hard problem, Harel and
Koren use a straightforward and efficient 2-approximation algorithm that relies on BreadthFirst Search [Hoc96]. The fine-scale vertex relocations are done using the Kamada-Kawai
approach. The Harel and Koren algorithm is summarized in Figure 12.5.


392 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS


Layout(G(V, E))
% Goal: Find L, a nice layout of G
% Constants:
% Rad[= 7] – determines radius of local neighborhoods
% Iterations[= 4] – determines number of iterations in local beautification
% Ratio[= 3] – ratio between number of vertices in two consecutive levels
% MinSize[= 10] – size of the coarsest graph
Compute the all-pairs shortest path length: dV V
Set up a random layout L
k ← MinSize
while k ≤|V | do
centers ←K-Centers(G(V, E), k)
radius = maxv∈centers minu∈centers{dvu} ∗ Rad
LocalLayout(dcenters×centers, L(centers), radius, Iterations)
for every v ∈ V do
L(v) ∈ L(center(v)) + rand
k ← kRatio
return L


K-Centers(G(V, E), k)
% Goal: Find a set S ⊆ V of size k, such that maxv∈V mins∈S{dsv} is
minimized.
S ←{v} for some arbitrary v ∈ V
for i = 2 to k do
1. Find the vertex u farthest away from S
(i.e., such that mins∈S{dus} ≥ mins∈S{dws}, ∀w ∈ V )
2. S ← S ∪{u}
return S


LocalLayout(dV ×V, L, k, Iterations)
% Goal: Find a locally nice layout L by beautifying k-neighborhoods
% dV ×V : all-pairs shortest path length
% L: initialized layout
% k: radius of neighborhoods
for i = 1 to Iterations ∗|V | do
1. Choose the vertex v with the maximal ∆ [k] v
2. Compute δv [k] [as in Kamada-Kawai]
3. L(v) ← L(v) + (δv [k][(][x][)][, δ] v [k][(][y][))]
end


Figure 12.5 Pseudo-code for the algorithm by Harel and Koren [HK02].


12.6. LARGE GRAPHS 393


Main Algorithm


create a filtration V : V0 ⊃ V1 ⊃ . . . ⊃ Vk ⊃∅
for i = k to 0 do
for each v ∈ Vi − Vi+1 do
find vertex neighborhood Ni(v), Ni−1(v), . . ., N0(v)
find initial position pos[v] of v
repeat rounds times
for each v ∈ Vi do
compute local temperature heat[v]
disp[v] ← heat[v] · [−→] FNi (v)
for each v ∈ Vi do
pos[v] ← pos[v] + disp[v]
add all edges e ∈ E


Figure 12.6 Pseudo-code for the algorithm by Gajer et al. [GGK04].


The 2000 algorithm of Gajer et al. [GGK04], shown in Figure 12.6, is also a multiscale force-directed algorithm but introduces several ideas to the realm of multi-scale forcedirected algorithms for large graphs. Most importantly, this approach avoids the quadratic
space and time complexity of previous force-directed approaches with the help of a simpler
coarsening strategy. Instead of computing a series of coarser graphs from the given large
graph G = (V, E), Gajer et al. produce a vertex filtration V : V0 ⊃ V1 ⊃ . . . ⊃ Vk ⊃∅,
where V0 = V (G) is the original vertex set of the given graph G. By restricting the number
of vertices considered in relocating any particular vertex in the filtration and ensuring that
the filtration has O(log |V |) levels an overall running time of O(|V | log [2] |V |) is achieved.
Filtrations based on graph centers (as in Harel and Koren [HK02]) and maximal independent
sets are considered. V = V0 ⊃ V1 ⊃ . . . ⊃ Vk ⊃∅, is a maximal independent set filtration
of G if Vi is a maximal subset of Vi−1 for which the graph distance between any pair of its
elements is greater than or equal to 2 [i] .
In the GRIP system [GK02], Gajer et al. add to the filtration and neighborhood calculations of [GGK04]: they introduce the idea of realizing the graph in high-dimensional
Euclidean space and obtaining 2D or 3D projections at the end. The algorithm also relies
on intelligent initial placement of vertices based on graph theoretic distances, rather than
on random initial placement. Finally, the notion of cooling is re-introduced in the context
of multi-scale force-directed algorithms. The GRIP system produces high-quality layouts, as
illustrated in Figure 12.7.
Another multilevel algorithm is that of Walshaw [Wal03]. Instead of relying on the
Kamada-Kawai type force interactions, this algorithm extends the grid variant of FruchtermanReingold to a multilevel algorithm. The coarsening step is based on repeatedly collapsing maximally independent sets of edges, and the fine-scale refinements are based on
Fruchterman-Reingold force calculations. This O(|V | [2] ) algorithm is summarized in Figure 12.8.


394 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS


The fourth 2000 multilevel force-directed algorithm is due to Quigley and Eades [QE00].
This algorithm relies on the Barnes-Hut n-body simulation method [BH86] and reduces
repulsive force calculations to O(|V | log |V |) time instead of the usual O(|V | [2] ). Similarly,
the algorithm of Hu [Hu05] combines the multilevel approach with the n-bosy simulation
method, and is implemented in the sfdp drawing engine of GraphViz [EGK [+] 01].
One possible drawback to this approach is that the running time depends on the distribution of the vertices. Hachul and J¨unger [HJ04] address this problem in their 2004 multilevel
algorithm.


Figure 12.7 Drawings from GRIP. First row: knotted meshes of 1600, 2500, and 10000
vertices. Second row: Sierpinski graphs of order 7 (1,095 vertices), order 6 (2,050 vertices),
3D Sierpinski of order 7 (8,194 vertices) [GK02].


12.6. LARGE GRAPHS 395


function fg(x, w):=begin return −Cwk [2] /x end
function fl(x, d, w):=begin return {(x − k)/d} − fg(x, w) end
t := t0;
Posn := NewPosn;
while (converged ̸= 1) begin
converged :=1;
for v ∈ V begin
OldPosn[v] = NewPosn[v]
end
for v ∈ V begin
{initialize D, the vector of displacements of v}
D := 0;
{calculate global (repulsive) forces}
for u ∈ V, u ̸= v begin
∆:= Posn[u] − Posn[v];
D := D + (∆/|Delta|) ∗ fg(|∆|, |u|);
end
{calculate local (spring) forces }
for u ∈ Γ(v) begin
∆:= Posn[u] − Posn[v];
D := D + (∆/|Delta|) ∗ fl(|∆|, |Γ(v)|, |u|);
end
{reposition v}
NewPosn[v] = NewPosn[v] + (D/|D|) ∗ min(t, |D|);
∆:= NewPosn[v] − OldPosn[v];
if (|∆| > k × tol)converged := 0;
end
{reduce the temperature to reduce the maximum movement}
t := cool(t);
end


Figure 12.8 Pseudo-code for the algorithm by Walshaw [Wal03].


396 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS

### 12.7 Stress Majorization


Methods that exploit fast algebraic operations offer another practical way to deal with
large graphs. Stress minimization has been proposed and implemented in the more general
setting of multidimensional scaling (MDS) [Kru64]. The function describing the stress is
similar to the layout energy function of Kamada-Kawai from Section 12.4:


�n


j=i+1


E =


n�−1


i=1


1
2 [k][i,j][(][|][p][i][ −] [p][j][| −] [l][i,j][)][2][,]


but here ki,j=1 and li,j = di,j is simply the graph theoretic distance. In their paper on
graph drawing by stress minimization Gansner et al. [GKN04] point out that this particular
formulation of the energy of the layout, or stress function has been already used to draw
graphs as early as in 1980 [KS80]. What makes this particular stress function relevant to
drawing large graphs is that it can be optimized better than with the local Newton-Raphson
method or with gradient descent. Specifically, this stress function can be globally minimized
via majorization. That is, unlike the energy function of Kamada-Kawai, the classical MDS
stress function can be optimized via majorization which is guaranteed to converge.
The strain model, or classical scaling, is related to the stress model. In this setting
a solution can be obtained via an eigen-decomposition of the adjacency matrix. Solving
the full stress or strain model still requires computing all pairs shortest paths. Significant
savings can be gained if we instead compute a good approximation. In PivotMDS Brandes
and Pich [BP06] show that replacing the all-pairs-shortest path computation with a distance
calculations from a few vertices in the graph is often sufficient, especially if combined with
a solution to a sparse stress model.
When not all nodes are free to move, constrained stress majorization can be used to
support additional constraints by, and treating the majorizing functions as a quadratic
program [DKM09]. Planar graphs are of particular interest in graph drawing, and often
force-directed graph drawing algorithms are used to draw them. While in theory any planar
graph has a straight-line crossings-free drawing in the plane, force-directed algorithms do
not guarantee such drawings.
Modifications to the basic force-directed functionality, with the aim of improving the layout quality for planar graphs, have also been considered. Harel and Sardas [HS98] improve
an earlier simulated annealing drawing algorithm by Davidson and Harel [DH96]. The main
idea is to obtain an initial plane embedding and then apply simulated annealing while not
introducing any crossings. Overall their method significantly improved the aesthetic quality
of the initial planar layouts, but at the expense of a significant increase in running time of
O(n [3] ), making it practical only for small graphs.
PrEd [Ber00] and ImPrEd [PSA11] are force-directed algorithms that improve already
created drawings of a graph. PrEd [Ber00] extends the method of Fruchterman and Reingold [FR91] and can be used as a post-processing crossings-preserving optimization. In
particular, PrEd takes some straight-line drawing as input and guarantees that no new
edge crossings will be created (while preserving existing crossings, if any are present in the
input drawing). Then the algorithm can be used to optimize a planar layout, while preserving its planarity and its embedding, or to improve a graph that has a meaningful initial set
of edge crossings. To achieve this result, PrEd adds a phase where the maximal movement
of each node is computed, and adds a repulsive force between (node, edge) pairs. The main
aims of ImPrEd [PSA11] are to significantly reduce the running time of PrEd, achieve high
aesthetics even for large and sparse graphs, and make the algorithm more stable and reliable


12.8. NON-EUCLIDEAN APPROACHES 397


with respect to the input parameters. This is achieved via improved spacing of the graph
elements and an accelerated convergence of the drawing to its final configuration.
An alternative approach for modifying force-directed functionality is to use a preprocessing step rather than a random layout to initialize the algorithm. Experimental results
indicate that combining a linear-time planar embedding step with a standard force-directed
algorithm such as a Fruchterman-Reingold can lead to improved qualitative and quantitative
results [FK12].

### 12.8 Non-Euclidean Approaches


Much of the work on non-Euclidean graph drawing has been done in hyperbolic space which
offers certain advantages over Euclidean space; see Munzner [Mun97, MB96]. For example,
in hyperbolic space it is possible to compute a layout for a complete tree with both uniform
edge lengths and uniform distribution of nodes. Furthermore, some of the embeddings of
hyperbolic space into Euclidean space naturally provide a fish-eye view of the space, which
is useful for “focus+context” visualization, as shown by Lamping et al. [LRP95]. From
a visualization point of view, spherical space offers a way to present a graph in a centerfree and periphery-free fashion. That is, in traditional drawings in R [2] there is an implicit
assumption that nodes in the center are important, while nodes on the periphery are less
important. This can be avoided in S [2] space, where any part of the graph can become
the center of the layout. The early approaches for calculating the layouts of graphs in
hyperbolic space, however, are either restricted by their nature to the layout of trees and
tree-like graphs, or to layouts on a lattice.
The hyperbolic tree layout algorithms function on the principle of hyperbolic sphere
packing, and operate by making each node of a tree, starting with the root, the center of a
sphere in hyperbolic space. The children of this node are then given positions on the surface
of this sphere and the process recurses on these children. By carefully computing the radii
of these spheres it is possible to create aesthetically pleasing layouts for the given tree.
Although some applications calculate the layout of a general graph using this method, the
layout is calculated using a spanning tree of the graph and the extra edges are then added
in without altering the layout [Mun98]. This method works well for tree-like and quasihierarchical graphs, or for graphs where domain-specific knowledge provides a way to create
a meaningful spanning tree. However, for general graphs (e.g., bipartite or densely connected
graphs) and without relying on domain specific knowledge, the tree-based approach may
result in poor layouts.
Methods for generalizing Euclidean geometric algorithms to hyperbolic space, although
not directly related to graph drawing, have also been studied. Recently, van Wijk and
Nuij [vWN04] proposed a Poincar´e’s half-plane projection to define a model for 2D viewing
and navigation. Eppstein [Epp03] shows that many algorithms that operate in Euclidean
space can be extended to hyperbolic space by exploiting the properties of a Euclidean model
of the space, such as the Beltrami-Klein or Poincar´e.
Hyperbolic and spherical space have also been used to display self-organizing maps in
the context of data visualization. Ontrup and Ritter [OR01] and Ritter [Rit99] extend the
traditional use of a regular (Euclidean) grid, on which the self-organizing map is created,
with a tessellation in spherical or hyperbolic space. An iterative process is then used to
adjust which elements in the data-set are represented by the intersections. Although the
hyperbolic space method seems to be a promising way to display high-dimensional data-sets,
the restriction to a lattice is often undesirable for graph visualization.


398 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS


Figure 12.9 Layouts of a graph obtained from research papers’ titles in hyperbolic space
H [2] and in spherical space S [2] [KW05].


Ostry [Ost96] considers constraining force-directed algorithms to the surface of threedimensional objects. This work is based on a differential equation formulation of the motion
of the nodes in the graph, and is flexible in that it allows a layout on almost any object,
even multiple objects. Since the force calculations are made in Euclidean space, however,
this method is inapplicable to certain geometries (e.g., hyperbolic geometry).
Another example of graph embedding within a non-Euclidean geometry is described in the
context of generating spherical parameterizations of 3D meshes. Gotsman et al. [GGS03]
describe a method for producing such an embedding using a generalization to spherical
space of planar methods for expressing convex combinations of points. The implementation
of the procedure is similar to the method described in this paper, but it may not lend itself
to geometries other than spherical.
Kobourov and Wampler [KW05] describe a conceptually simple approach to generalizing
force-directed methods for graph layout from Euclidean geometry to Riemannian geometries. Unlike previous work on non-Euclidean force-directed methods, this approach is
not limited to special classes of graphs but can be applied to arbitrary graphs; see Figure 12.9. The method relies on extending the Euclidean notions of distance, angle, and
force-interactions to smooth non-Euclidean geometries via projections to and from appropriately chosen tangent spaces. Formal description of the calculations needed to extend
such algorithms to hyperbolic and spherical geometries are also detailed.
In 1894 Riemann described a generalization of the geometry of surfaces, which had been
studied earlier by Gauss, Bolyai, and Lobachevsky. Two well-known special cases of Riemannian geometries are the two standard non-Euclidean types, spherical geometry and
hyperbolic geometry. This generalization led to the modern concept of a Riemannian manifold. Riemannian geometries have less convenient structure than Euclidean geometry, but
they do retain many of the characteristics which are useful for force-directed graph layouts.
A Riemannian manifold M has the property that for every point x ∈ M, the tangent space
TxM is an inner product space. This means that for every point on the manifold, it is
possible to define local notions of length and angle.


12.8. NON-EUCLIDEAN APPROACHES 399


Using the local notions of length we can define the length of a continuous curve γ : [a, b] →
M by


                  - b
length(γ) = ||γ [′] (t)||dt.

a


This leads to a natural generalization of the concept of a straight line to that of a geodesic,
where the geodesic between two points u, v ∈ M is defined as a continuously differentiable
curve of minimal length between them. These geodesics in Euclidean geometry are straight
lines, and in spherical geometry they are arcs of great circles.
We can similarly define the distance between two points, d(x, y) as the length of a geodesic
between them. In Euclidean space the relationship between a pair of nodes is defined along
lines: the distance between the two nodes is the length of the line segment between them
and forces between the two nodes act along the line through them. These notions of distance
and forces can be extended to a Riemannian geometry by having these same relationships
be defined in terms of the geodesics of the geometry, rather than in terms of Euclidean lines.
As Riemannian manifolds have a well-structured tangent space at every point, these tangent spaces can be used to generalize spring embedders to arbitrary Riemannian geometries.
In particular, the tangent space is useful in dealing with the interaction between one point
and several other points in non-Euclidean geometries. Consider three points x, y, and z in
a Riemannian manifold M where there is an attractive force from x to y and z. As can
be easily seen in the Euclidean case (but also true in general) the net force on x is not
necessarily in the direction of y or z, and thus the natural motion of x is along neither the
geodesic toward y, nor that toward z. Determining the direction in which x should move
requires the notion of angle.
Since the tangent space at x, being an inner product space, has enough structure to define
lengths and angles, we do the computations for calculating the forces on x in TxM . In order
to do this, we define two functions for every point x ∈ M as follows:


τx : M → TxM


τx [−][1] : TxM → M


These two functions map points in M to and from the tangent space of M at x, respectively. We require that τx and τx [−][1] satisfy the following constraints:


1. τx [−][1][(][τ][x][(][y][)) =][ y][ for all][ y][ ∈] [M]
2. ||τx(y)|| = d(x, y)
3. τx preserves angles about the origin


Using these functions it is now easy to define the way in which the nodes of a given
graph G = (V, E) interact with each other through forces. In the general framework for this
algorithm each node is considered individually, and its new position is calculated based on
the relative locations of the other nodes in the graph (repulsive forces) and on its adjacent
edges (attractive forces). Then we obtain pseudo-code for a traditional Euclidean spring
embedder and its corresponding non-Euclidean counterpart, as shown in Figure 12.10.


400 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS


generic ~~a~~ lgorithm(G)
while not done do
foreach n ∈ G do
position[n] := force ~~d~~ irected ~~p~~ lacement(n, G)
end
non ~~E~~ uclidean algorithm(G)
while not done do
foreach n ∈ G do
x := position[n]
G [′] := τx(G)
x [′] := force ~~d~~ irected ~~p~~ lacement(n, G [′] )
position[n] := τx [−][1][(][x][′][)]
end
end


Figure 12.10 Pseudo-code for a traditional Euclidean spring embedder and its corresponding non-Euclidean counterpart.

### 12.9 Lombardi Spring Embedders


Inspired by American graphic artist Mark Lombardi, Duncan et al. [DEG [+] 10a, DEG [+] 10b]
introduce the concept of a Lombardi drawing, which is a drawing that uses circular arcs
for edges and achieves the maximum (i.e., perfect) amount of angular resolution possible at
each vertex.


There are several force-directed graph drawing methods that use circular-arc edges or
curvilinear poly-edges. Brandes and Wagner [BW00] describe a force-directed method for
drawing train connections, where the vertex positions are fixed but transitive edges are
drawn as B´ezier curves. Finkel and Tamassia [FT05], on the other hand, describe a forcedirected method for drawing graphs using curvilinear edges where vertex positions are free
to move. Their method is based on adding dummy vertices that serve as control points for
B´ezier curve.

Chernobelskyi et al. [CCG [+] 11] describe two force-directed algorithms for Lombardi-style
(or near-Lombardi ) drawings of graphs, where edges are drawn using circular arcs with the
goal of maximizing the angular resolution at each vertex. The first approach calculates
lateral and rotational forces based on the two tangents defining a circular arc between two
vertices. In contrast, the second approach uses dummy vertices on each edge with repulsive
forces to “push out” the circular arcs representing edges, so as to provide an aesthetic
“balance”. Another distinction between the two approaches is that the first one lays out
the vertex positions along with the circular edges, while the second one works on graphs
that are already laid out, only modifying the edges. It can be argued that Lombardi or
near-Lombardi graph drawings have a certain aesthetic appeal as has been shown in recent
empirical experiments [PHNK12]; see Fig. 12.11. However, another recent experimental
paper on curve-based drawings [XRP [+] 12] seems to suggest that straight-line drawings have
better readability.


12.10. DYNAMIC GRAPH DRAWING 401


Figure 12.11 Examples of force-directed Lombardi drawings: note that every edge is a
circular arc and every vertex has perfect angular resolution [CCG [+] 11].

### 12.10 Dynamic Graph Drawing


While static graphs arise in many applications, dynamic processes give rise to graphs that
evolve through time. Such dynamic processes can be found in software engineering, telecommunications traffic, computational biology, and social networks, among others.


Thus, dynamic graph drawing deals with the problem of effectively presenting relationships as they change over time. A related problem is that of visualizing multiple relationships
on the same dataset. Traditionally, dynamic relational data is visualized with the help of
graphs, in which vertices and edges fade in and out as needed, or as a time-series of graphs;
see Figure 12.12.


Figure 12.12 A dynamic graph can be interpreted as a larger graph made of connecting
graphs in adjacent timeslices [EHK [+] 04].


402 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS


The input to this problem is a series of graphs defined on the same underlying set
of vertices. As a consequence, nearly all existing approaches to visualization of evolving and dynamic graphs are based on the force-directed method. Early work can be
dated back to North’s DynaDAG [Nor96], where the graph is not given all at once, but
incrementally. Brandes and Wagner adapt the force-directed model to dynamic graphs
using a Bayesian framework [Brandes and Wagner 1998]. Diehl and G¨org [DG02] consider graphs in a sequence to create smoother transitions. Special classes of graphs such
as trees, series-parallel graphs and st-graphs have also been studied in dynamic models [CDTT95, CBT [+] 92, Moe90]. Most of these early approaches, however, are limited
to special classes of graphs and usually do not scale to graphs over a few hundred vertices.
TGRIP was one of the first practical tools that could handle the larger graphs that appear
in the real-world. It was developed as part of a system that keeps track of the evolution of
software by extracting information about the program stored within a CVS version control
system [CKN [+] 03]. Such tools allow programmers to understand the evolution of a legacy
program: Why is the program structured the way it is? Which programmers were responsible for which parts of the program during which time periods? Which parts of the program
appear unstable over long periods of time? TGRIP was used to visualize inheritance graphs,
program call-graphs, and control-flow graphs, as they evolve over time; see Fig. 12.13.
For layout of evolving and dynamic graphs, there are two important criteria to consider:


1. readability of the individual layouts, which depends on aesthetic criteria such as
display of symmetries, uniform edge lengths, and minimal number of crossings;
and
2. mental map preservation in the series of layouts, which can be achieved by ensuring that vertices and edges that appear in consecutive graphs in the series,
remain in the same location.


These two criteria are often contradictory. If we obtain individual layouts for each graph,
without regard to other graphs in the series, we may optimize readability at the expense of
mental map preservation. Conversely, if we fix the common vertices and edges in all graphs
once and for all, we are optimizing the mental map preservation yet the individual layouts
may be far from readable. Thus, we can measure the effectiveness of various approaches for
visualization of evolving and dynamic graphs by measuring the readability of the individual
layouts, and the overall mental map preservation.


Figure 12.13 Snapshots of the call-graph of a program as it evolves through time,
extracted from CVS logs. Vertices start out red. As time passes and a vertex does not
change it turns purple and finally blue. When another change is affected, the vertex again
becomes red. Note the number of changes between the two large clusters and the break in
the build on the last image [CKN [+] 03].


12.11. CONCLUSION 403


Dynamic graphs can be visualized with aggregated views, where all the graphs are displayed at once, merged views, where all the graphs are stacked above each other, and with
animations, where only one graph is shown at a time, and morphing is used when changing between graphs (fading in/out vertices and edges that appear/disappear). When using
the animation/morphing approach, it is possible to change the balance between readability of individual graphs and the overall mental map preservation, as in the system for
Graph Animations with Evolving Layouts, GraphAEL [EHK [+] 03, FKN [+] 04]. Applications
of this framework include visualizing software evolution [CKN [+] 03], social networks analysis [MB09], and the behavior of dynamically modifiable code [DID [+] 05].

### 12.11 Conclusion


Force-directed algorithms for drawing graphs have a long history and new variants are still
introduced every year. Their intuitive simplicity appeals to researchers from many different
fields, and this accounts for dozens of available implementations. As new relational data
sets continue to be generated in many applications, force-directed algorithms will likely
continue to be the method of choice. The latest scalable algorithms and algorithms that
can handle large dynamic and streaming graphs are arguably of greatest utility today.


404 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS

### References


[BBS97] J¨urgen Branke, Frank Bucher, and Hartmut Schmeck. A genetic algorithm
for drawing undirected graphs. In Proceedings of the 3rd Nordic Workshop
on Genetic Algorithms and Their Applications, pages 193–206, 1997.

[Ber00] Francois Bertault. A Force-Directed Algorithm that Preserves Edge Crossing Properties. Information Processing Letters, 74(1-2):7–13, 2000.

[BF96] I. Bruß and A. Frick. Fast interactive 3-D graph visualization. In F. J.
Brandenburg, editor, Proceedings of the 3rd Symposium on Graph Drawing
(GD), volume 1027 of Lecture Notes Computer Science, pages 99–110.
Springer-Verlag, 1996.

[BH86] Josh Barnes and Piet Hut. A hierarchical O(N log N) force calculation
algorithm. Nature, 324:446–449, December 1986.

[BP06] U. Brandes and C. Pich. Eigensolver methods for progressive multidimensional scaling of large data. In Proceedings 14th Symposium on Graph
Drawing (GD), pages 42–53, 2006.

[Bra01] Ulrik Brandes. Drawing on physical analogies. In Michael Kaufmann and
Dorothea Wagner, editors, Drawing Graphs, volume 2025 of Lecture Notes
in Computer Science, pages 71–86. Springer-Verlag, 2001.

[BW00] Ulrik Brandes and Dorothea Wagner. Using Graph Layout to Visualize
Train Interconnection Data. J. Graph Algorithms Appl., 4(3):135–155,
2000.

[CBT [+] 92] R. F. Cohen, G. Di Battista, R. Tamassia, I. G. Tollis, and P. Bertolazzi.
A framework for dynamic graph drawing. In Proceedings of the 8th Annual
Symposium on Computational Geometry (SCG ’92), pages 261–270, 1992.

[CCG [+] 11] R. Chernobelskiy, K. Cunningham, M. T. Goodrich, S. G. Kobourov, and
L. Trott. Force-directed lombardi-style graph drawing. In Proceedings
19th Symposium on Graph Drawing (GD), pages 78–90, 2011.

[CDTT95] R. F. Cohen, G. Di Battista, R. Tamassia, and I. G. Tollis. Dynamic
graph drawings: Trees, series-parallel digraphs, and planar ST -digraphs.
SIAM J. Comput., 24(5):970–1001, 1995.

[CKN [+] 03] C. Collberg, S. G. Kobourov, J. Nagra, J. Pitts, and K. Wampler. A
system for graph-based visualization of the evolution of software. In ACM
Symposium on Software Visualization (SoftVis), pages 77–86, 2003.

[CLRS90] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein. Introduction
to Algorithms. MIT Press, Cambridge, MA, 1990.

[Coh97] Jonathan D. Cohen. Drawing graphs to convey proximity: An incremental
arrangement method. ACM Transactions on Computer-Human Interaction, 4(3):197–229, September 1997.

[CT96] I. F. Cruz and J. P. Twarog. 3D graph drawing with simulated annealing.
In F. J. Brandenburg, editor, Proceedings of the 3rd Symposium on Graph
Drawing (GD), volume 1027 of Lecture Notes Computer Science, pages
162–165. Springer-Verlag, 1996.

[DEG [+] 10a] Christian A. Duncan, David Eppstein, Michael T. Goodrich, Stephen G.
Kobourov, and Martin N¨ollenburg. Drawing trees with perfect angular
resolution and polynomial area. In Graph Drawing, pages 183–194, 2010.


REFERENCES 405


[DEG [+] 10b] Christian A. Duncan, David Eppstein, Michael T. Goodrich, Stephen G.
Kobourov, and Martin N¨ollenburg. Lombardi drawings of graphs. In
Graph Drawing, pages 195–207, 2010.

[DETT99] Giuseppe Di Battista, Peter Eades, Roberto Tamassia, and Ioannis G. Tollis. Graph Drawing: Algorithms for the Visualization of Graphs. Prentice
Hall, Englewood Cliffs, NJ, 1999.

[DG02] Stephan Diehl and Carsten G¨org. Graphs, they are changing. In Proceedings of the 10th Symposium on Graph Drawing (GD), pages 23–30,
2002.

[DH96] Ron Davidson and David Harel. Drawing graphs nicely using simulated
annealing. ACM Transactions on Graphics, 15(4):301–331, 1996.

[DID [+] 05] Brad Dux, Anand Iyer, Saumya Debray, David Forrester, and Stephen G.
Kobourov. Visualizing the behaviour of dynamically modifiable code. In
13th IEEE Workshop on Porgram Comprehension, pages 337–340, 2005.

[DKM09] Tim Dwyer, Yehuda Koren, and Kim Marriott. Constrained graph layout
by stress majorization and gradient projection. Discrete Mathematics,
309(7):1895–1908, 2009.

[Ead84] Peter Eades. A heuristic for graph drawing. Congressus Numerantium,
42:149–160, 1984.

[EG95] Peter Eades and Patrick Garvan. Drawing stressed planar graphs in three
dimensions. In Proceedings of the 3rd Symposium on Graph Drawing,
pages 212–223, 1995.

[EGK [+] 01] John Ellson, Emden R. Gansner, Eleftherios Koutsofios, Stephen C.
North, and Gordon Woodhull. Graphviz—open source graph drawing
tools. In Graph Drawing, pages 483–484, 2001.

[EHK [+] 03] C. Erten, P. J. Harding, S. G. Kobourov, K. Wampler, and G. Yee.
GraphAEL: Graph animations with evolving layouts. In 11th Symposium
on Graph Drawing, pages 98–110, 2003.

[EHK [+] 04] C. Erten, P. J. Harding, S. Kobourov, K. Wampler, and G. Yee. Exploring the computing literature using temporal graph visualization. In
Visualization and Data Analysis, pages 45–56, 2004.

[Epp03] D. Eppstein. Hyperbolic geometry, M¨obius transformations, and geometric optimization. In MSRI Introductory Workshop on Discrete and Computational Geometry, 2003.

[FCW67] C. Fisk, D. Caskey, and L. West. Accel: Automated circuit card etching
layout. Proceedings of the IEEE, 55(11):1971–1982, 1967.

[FK12] Joe Fowler and Stephen G. Kobourov. Planar preprocessing for spring
embedders. In Graph Drawing, 2012.

[FKN [+] 04] D. Forrester, S. G. Kobourov, A. Navabi, K. Wampler, and G. Yee.
graphael: A system for generalized force-directed layouts. In 12th Symposium on Graph Drawing (GD), 2004.

[FLM95] A. Frick, A. Ludwig, and H. Mehldau. A fast adaptive layout algorithm for
undirected graphs. In R. Tamassia and I. G. Tollis, editors, Proceedings
of the 2nd Symposium on Graph Drawing (GD), volume 894 of Lecture
Notes in Computer Science, pages 388–403. Springer-Verlag, 1995.

[FR91] T. Fruchterman and E. Reingold. Graph drawing by force-directed placement. Softw. – Pract. Exp., 21(11):1129–1164, 1991.


406 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS


[FT05] Benjamin Finkel and Roberto Tamassia. Curvilinear Graph Drawing Using the Force-Directed Method. In Proc. 12th Int. Symp. on Graph Drawing (GD 2004), pages 448–453, 2005.

[GGK04] P. Gajer, M. T. Goodrich, and S. G. Kobourov. A fast multi-dimensional
algorithm for drawing large graphs. Computational Geometry: Theory
and Applications, 29(1):3–18, 2004.

[GGS03] C. Gotsman, X. Gu, and A. Sheffer. Fundamentals of spherical parameterization for 3D meshes. In ACM Transactions on Graphics, 22, pages
358–363, 2003.

[GK02] Pawel Gajer and Stephen G. Kobourov. GRIP: Graph dRawing with
Intelligent Placement. Journal of Graph Algorithms and Applications,
6(3):203–224, 2002.

[GKN04] E. Gansner, Y. Koren, and S. North. Graph drawing by stress minimization. In Proceedings 12th Symposium on Graph Drawing (GD), pages
239–250, 2004.

[Gre88] Leslie Greengard. The Rapid Evolution of Potential Fields in Particle
Systems. MIT. Press, Cambridge, MA, 1988.

[HH01] R. Hadany and D. Harel. A multi-scale algorithm for drawing graphs
nicely. Discrete Applied Mathematics, 113(1):3–21, 2001.

[HJ04] S. Hachul and M. J¨unger. Drawing large graphs woth a potential-fieldbased multilevel algorithm. In Proceedings of the 12th Symposium on
Graph Drawing (GD), volume 3383 of Lecture Notes in Computer Science,
pages 285–295. Springer-Verlag, 2004.

[HK02] David Harel and Yehuda Koren. A fast multi-scale method for drawing
large graphs. Journal of Graph Algorithms and Applications, 6(3):179–
2002, 2002.

[Hoc96] D. S. Hochbaum. Approximation Algorithms for NP-Hard Problems. PWS
Publishing, 1996.

[HS98] David Harel and Meir Sardas. An algorithm for straight-line drawing of
planar graphs. Algorithmica, 20(2):119–135, 1998.

[Hu05] Yifan Hu. Efficient and high quality force-directed graph drawing. The
Mathematica Journal, 10:37–71, 2005.

[KGV83] S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi. Optimization by simulated
annealing. Science, 220(4598):671–680, 1983.

[KK89] T. Kamada and S. Kawai. An algorithm for drawing general undirected
graphs. Inform. Process. Lett., 31:7–15, 1989.

[KMS91] Corey Kosak, Joe Marks, and Stuart Shieber. A parallel genetic algorithm for network-diagram layout. In Proceedings of the 4th International
Conference on Genetic Algorithms, pages 458–465, 1991.

[Kru64] J. B. Kruskal. Multidimensional scaling by optimizing goodness of fit to
a nonmetric hypothesis. Psychometrika, 29:1–27, 1964.

[KS80] J. Kruskal and J. Seery. Designing network diagrams. In Proceedings 1st
General Conference on Social Graphics, pages 22–50, 1980.

[KW05] S. G. Kobourov and K. Wampler. Non-Euclidean spring embedders. IEEE
Transactions on Visualization and Computer Graphics, 11(6):757–767,
2005.


REFERENCES 407


[LR88] T. Leighton and S. Rao. An approximate max-flow min-cut theorem for
uniform multicommodity flow problems with applications to approximation algorithms. In Proceedings of the 29th Annual Symposium on Foundations of Computer Science (FOCS), pages 422–431, 1988.

[LRP95] John Lamping, Ramana Rao, and Peter Pirolli. A focus+context technique based on hyperbolic geometry for visualizing large hierarchies. In
Proceedings of Computer Human Interaction, pages 401–408. ACM, 1995.

[MB96] T. Munzner and P. Burchard. Visualizing the structure of the World Wide
Web in 3D hyperbolic space. In David R. Nadeau and John L. Moreland, editors, 1995 Symposium on the Virtual Reality Modeling Language,
VRML ’95, pages 33–38, 1996.

[MB09] M. Jacomy M. Bastian, S. Heymann. Gephi: an open source software for
exploring and manipulating networks. International AAAI Conference on
Weblogs and Social Media, 2009.

[Moe90] Sven Moen. Drawing dynamic trees. IEEE Software, 7(4):21–28, July
1990.

[Mun97] Tamara Munzner. H3: Laying out large directed graphs in 3D hyperbolic space. In L. Lavagno and W. Reisig, editors, Proceedings of IEEE
Symposium on Information Visualization, pages 2–10, 1997.

[Mun98] T. Munzner. Drawing large graphs with H3Viewer and Site Manager.
In Proceedings of the 6th Symposium on Graph Drawing, pages 384–393,
1998.

[Noa07] Andreas Noack. Energy models for graph clustering. J. Graph Algorithms
Appl., 11(2):453–480, 2007.

[Nor96] S. C. North. Incremental layout in DynaDAG. In Proceedings of the 4th
Symposium on Graph Drawing (GD), pages 409–418, 1996.

[OR01] J. Ontrup and H. Ritter. Hyperbolic self-organizing maps for semantic
navigation. In Advances in Neural Information Processing Systems 14,
pages 1417–1424, 2001.

[Ost96] Diethelm Ironi Ostry. Some three-dimensional graph drawing algorithms.
Master’s thesis, University of Newcastle, Australia, 1996.

[PHNK12] Helen Purchase, John Hamer, Martin N¨ollenburg, and Stephen G.
Kobourov. On the usability of Lombardi graph drawings. In Graph Drawing, 2012.

[PSA11] Daniel Archambault Paolo Simonetto and David Auber. ImPrEd: An improved force-directed algorithm that prevents nodes from crossing edges.
Computer Graphics Forum (EuroVis), 30(3):1071–1080, 2011.

[QB79] N. Quinn and M. Breur. A force directed component placement procedure
for printed circuit boards. IEEE Transactions on Circuits and Systems,
CAS-26(6):377–388, 1979.

[QE00] Aaron Quigley and Peter Eades. FADE: graph drawing, clustering, and
visual abstraction. In Proceedings of the 8th Symposium on Graph Drawing
(GD), volume 1984 of Lecture Notes in Computer Science, pages 197–210.
Springer-Verlag, 2000.

[Rit99] H. Ritter. Self-organizing maps on non-euclidean spaces. In Erkki Oja
and Samuel Kaski, editors, Kohonen Maps, pages 97–110. Elsevier, Amsterdam, 1999.


408 CHAPTER 12. FORCE-DIRECTED DRAWING ALGORITHMS


[SM00] J. Shi and J. Malik. Normalized cuts and image segmentation. IEEE
Transaction on Pattern Analysis and Machine Intelligence, 22(8):888–905,
2000.

[Tut63] William T. Tutte. How to draw a graph. Proc. London Math. Society,
13(52):743–768, 1963.

[Vos99] Michael D. Vose. The Simple Genetic Algorithm: Foundations and Theory.
MIT Press, 1999.

[vWN04] J. J. van Wijk and W. A. A. Nuij. A model for smooth viewing and navigation of large 2D information spaces. IEEE Transactions on Visualization
and Computer Graphics, 10(4):447– 458, 2004.

[Wal03] C. Walshaw. A multilevel algorithm for force-directed graph drawing.
Journal of Graph Algorithms and Applications, 7(3):253–285, 2003.

[XRP [+] 12] K. Xu, C. Rooney, P. Passmore, D. H. Ham, and P. Nguyen. A user study
on curved edges in graph visualization. In IEEE InfoVis, 2012.
