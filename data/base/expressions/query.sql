SELECT nodetype_id as expressions__node_id,
       expression_category as expressions__syntactic_category,
       firstchild as expressions__syntactic_category_first_child,
       secondchild as expressions__syntactic_category_second_child,
       thirdchild as expressions__syntactic_category_third_child,
       parentnode as expressions__syntactic_category_parent_node,
       parentchild as expressions_role_played_in_parent_node,
       height as expressions__height,
       depth as expressions__depth,
       user_id as expressions__user_id,
       user_class as expressions_user_class

FROM expression