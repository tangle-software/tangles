{{ title }}

.. currentmodule:: {{ module }}

.. automodule:: {{ module }}
{% if sub_mods %}
   .. rubric:: Modules

   .. toctree::
      :hidden:
{% for sp, _, _ in sub_mods %}
      {{ sp }}
{%- endfor %}

   .. list-table::
      :widths: 50 50
{% for _, sub_mod, desc in sub_mods %}
      * - :mod:`{{module}}.{{sub_mod}}`
        - {{desc or ""}}
{%- endfor %}
{% endif %}
{% if classes %}
   .. rubric:: Classes

   .. toctree::
      :hidden:
{% for cp, _, _ in classes %}
      {{ cp }}
{%- endfor %}

   .. list-table::
      :widths: 50 50
{% for _, cls, desc in classes %}
      * - :class:`{{cls}}`
        - {{desc or ""}}
{%- endfor %}
{% endif %}
{% if functions %}
   .. rubric:: Functions

   .. toctree::
      :hidden:
{% for fp, _, _ in functions %}
      {{ fp }}
{%- endfor %}

   .. list-table::
      :widths: 50 50
{% for _, func, desc in functions %}
      * - :func:`{{func}}`
        - {{desc or ""}}
{%- endfor %}
{% endif %}