SELECT programs.program_id                   as program__program_id,
       programs.classpercentage              as programs__percentage_of_classes,
       programs.interfacepercentage          as programs__percentage_of_interfaces,
       programs.enumpercentage               as programs__percentage_of_enums,
       programs.codeinpackages               as programs__contains_code_in_packages,
       programs.defaultpackage               as programs__contains_code_in_default_package,
       type_defs.nodetype_id                 as type_def__node_id,
       type_defs.typenodetype                as type_def__syntactic_category,
       type_defs.typepublicvisibility        as type_def__visibility,
       type_defs.isfinalclass                as type_def__is_final,
       type_defs.hasextends                  as type_def__is_extension,
       type_defs.numberannotations           as type_def__number_of_annotations,
       type_defs.isindefaultpackage          as type_def__is_in_default_package,
       type_defs.numberimplements            as type_def__number_of_implements,
       type_defs.numbergenerictypes          as type_def__number_of_generics,
       type_defs.numbermethods               as type_def__number_of_methods,
       type_defs.percentageoverloadedmethods as type_def__percentage_overloaded_methods,
       type_defs.numberconstructors          as type_def__number_of_constructors,
       type_defs.numberfields                as type_def__number_of_fields,
       type_defs.numbernestedtypes           as type_def__number_of_nested_types,
       type_defs.numberinnertypes            as type_def__number_of_inner_types,
       type_defs.namingconvention            as type_def__naming_convention,
       type_defs.isabstract                  as type_def__is_abstract,
       type_defs.numberstaticnestedtypes     as type_def__number_of_static_nested_types,
       type_defs.isinnerclass                as type_def__is_inner,
       type_defs.isstrictfp                  as type_def__is_strictfp,
       type_defs.staticfieldpercentage       as type_def__percentage_of_static_fields,
       type_defs.staticmethodpercentage      as type_def__percentage_of_static_methods,
       type_defs.numberstaticblocks          as type_def__number_of_static_blocks,
       type_defs.isnestedclass               as type_def__is_nested,
       type_defs.isstatic                    as type_def__is_static,
       type_defs.user_id                     as type_defs__user_id,
       type_defs.user_class                  as type_def__user_class

FROM program programs,
     file files,
     typedefinition type_defs
WHERE programs.program_id = files.program_id
  AND files.file_id = type_defs.file_id;