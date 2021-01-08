USE spotify
CALL gds.graph.drop('myGraph', false);

USE spotify
CALL gds.graph.create('myGraph', 'ARTIST', '*', {relationshipProperties: 'weight'});

USE spotify
CALL gds.pageRank.stream('myGraph', {
  maxIterations: 20,
  dampingFactor: 0.85,
  relationshipWeightProperty: 'weight'
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, gds.util.asNode(nodeId).url AS url, score
ORDER BY score DESC, name ASC;
