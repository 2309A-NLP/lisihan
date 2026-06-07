# Neo4j 图谱查看查询

不要直接执行 `MATCH (n) RETURN n`，那会把全库节点堆在一起。建议按公司或问题实体查看子图。

## 查看某家公司一跳子图

```cypher
MATCH p=(c:Company)-[r]-(n)
WHERE c.name CONTAINS '平安银行'
RETURN p
LIMIT 80;
```

## 查看组织/部门关系

```cypher
MATCH p=(c:Company)-[r:`设有`|`调整`]-(n)
WHERE c.name CONTAINS '国泰君安'
RETURN p
LIMIT 80;
```

## 查看人员管理关系

```cypher
MATCH p=(c:Company)-[r:`人员`]-(n:Person)
WHERE c.name CONTAINS '平安银行'
RETURN p
LIMIT 50;
```

## 查看财务指标数值

```cypher
MATCH p=(c:Company)-[r:`指标`|`增长`]-(n)
WHERE c.name CONTAINS '招商证券'
RETURN p
LIMIT 80;
```

## 查看节点类型数量

```cypher
MATCH (n:GraphEntity)
RETURN n.type AS type, count(*) AS count
ORDER BY count DESC;
```
