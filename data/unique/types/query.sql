SELECT type_category as type__syntactic_category,
       primitive     as type__is_primitive,
       height        as type__height,
       parentnode    as type__syntactic_category_parent_node,
       parentchild   as type__role_played_in_parent_node,
       generics      as type__number_of_generics,
       dimensions    as type__number_of_dimensions,
       user_class    as type___user_class

FROM type