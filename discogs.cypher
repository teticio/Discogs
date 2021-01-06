LOAD CSV WITH HEADERS FROM
"file:///home/teticio/ML/Discogs/artists.csv" AS csv
MERGE (artist:Artist {name: csv.Name, id: csv.Id, profile: csv.Profile})

WITH artist
LOAD CSV WITH HEADERS FROM
"file:///home/teticio/ML/Discogs/releases.csv" AS csv
MERGE (release:Release {name: csv.Name, id: csv.Id, url: csv.URL})

WITH artist, release
LOAD CSV WITH HEADERS FROM
"file:///home/teticio/ML/Discogs/a2a.csv" AS csv
MATCH (p1:Artist {id: csv.FromId})
MATCH (p2:Artist {id: csv.ToId})
FOREACH(ignoreMe IN CASE WHEN csv.Relationship = "AKA" THEN [1] ELSE [] END |
 MERGE (p1)-[:AKA]->(p2))
FOREACH(ignoreMe IN CASE WHEN csv.Relationship = "PLAYED_IN" THEN [1] ELSE [] END |
 MERGE (p1)-[:PLAYED_IN]->(p2))
FOREACH(ignoreMe IN CASE WHEN csv.Relationship = "PLAYED_WITH" THEN [1] ELSE [] END |
 MERGE (p1)-[:PLAYED_WITH]->(p2))

WITH p1, p2
LOAD CSV WITH HEADERS FROM
"file:///home/teticio/ML/Discogs/a2r.csv" AS csv
MATCH (p3:Artist {id: csv.FromId})
MATCH (p4:Release {id: csv.ToId})
MERGE (p3)-[:APPEARS_ON]->(p4)

WITH p1, p2, p3, p4
LOAD CSV WITH HEADERS FROM
"file:///home/teticio/ML/Discogs/r2a.csv" AS csv
MATCH (p5:Release {id: csv.FromId})
MATCH (p6:Artist {id: csv.ToId})
MERGE (p5)-[:FEATURES {as: csv.Role}]->(p6)
;
