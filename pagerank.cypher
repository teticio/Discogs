USE discogs
CALL gds.graph.drop('myGraph', false);

USE discogs
CALL gds.graph.create('myGraph', 'ARTIST', '*', {});

USE discogs
CALL gds.pageRank.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC, name ASC;
