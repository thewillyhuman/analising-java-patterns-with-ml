SELECT statement_category as statements__syntactic_category,
       firstchild         as statements__syntactic_category_first_child,
       secondchild        as statements__syntactic_category_second_child,
       thirdchild         as statements__syntactic_category_third_child,
       parentnode         as statements__syntactic_category_parent_node,
       height             as statements__height,
       depth              as statements__depth,
       user_class         as statements__user_class

FROM statement