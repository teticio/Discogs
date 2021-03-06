USE discogs
CALL gds.graph.drop('myGraph', false);

USE discogs
CALL gds.graph.create('myGraph', 'ARTIST', '*', {});

USE discogs
CALL gds.pageRank.stream('myGraph', {
  maxIterations: 20,
  dampingFactor: 0.85
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, gds.util.asNode(nodeId).url AS url, score
ORDER BY score DESC, name ASC;
