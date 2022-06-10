SELECT fieldvisibility as field_def__visibility,
       isfinal as field_def__is_final,
       isstatic as field_def__is_static,
       numberannotations as field_def__number_of_annotations,
       namingconvention as field_def__naming_convention,
       initialvalue as field_def__initial_value,
       type as field_def__syntatic_category,
       isvolatile as field_def__is_volatile,
       istransient as field_def__is_transient,
       parentnode as field_def__syntactic_category_parent_node,
       user_class as field_def__user_class
FROM fielddefinition;