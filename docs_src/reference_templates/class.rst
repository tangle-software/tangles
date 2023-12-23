{{ title }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ name }}

{% if properties %}
   .. rubric:: Properties

   .. toctree::
      :hidden:
{% for pp, _, _ in properties %}
      {{ pp }}
{%- endfor %}

   .. list-table::
      :widths: 50 50
{% for _, property, desc in properties %}
      * - :attr:`{{property}}`
        - {{desc or ""}}
{%- endfor %}
{% endif %}

{% if methods %}
   .. rubric:: Methods

   .. toctree::
      :hidden:
{% for mp, _, _ in methods %}
      {{ mp }}
{%- endfor %}

   .. list-table::
      :widths: 50 50
{% for _, method, desc in methods %}
      * - :func:`{{method}}`
        - {{desc or ""}}
{%- endfor %}
{% endif %}