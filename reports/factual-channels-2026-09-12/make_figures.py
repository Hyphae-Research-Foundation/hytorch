#!/usr/bin/env python3
"""Render the technical report figures from its portable JSON only.

No extraction, model, scorer or statistical-inference operation is performed.
PDFs contain vector graphics/text; PNGs are previews of the same figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np

WIDTH = 166/25.4
NAVY = '#19364F'
TEAL = '#007F83'
AMBER = '#D18B13'
SLATE = '#718BA3'
GREY = '#62717E'
LIGHT = '#E4E9ED'
POSITIONS = ('identity', 'heldout-shift8', 'heldout-gaps4')
POSITION_SHORT = ('I', 'S8', 'G4')
POSITION_COLORS = (NAVY, TEAL, AMBER)
DIVERGING = LinearSegmentedColormap.from_list('navy_white_teal', [NAVY, '#F7F8F8', TEAL])
LABELS = {
    'es': {
        'competence_common': 'A · Common', 'competence_overall': 'B · Total',
        'competence_rare': 'C · Rare', 'competence_unexposed': 'D · No expuestos',
        'competence_title': '{label}\n{facts} hechos · {queries} preguntas',
        'gate': 'gate 80%', 'accuracy': 'Exactitud estricta (%)',
        'positions_note': 'I: identidad   ·   S8: desplazamiento +8   ·   G4: huecos 4',
        'competence_note': 'El gate se aplica a common; el total incluye los 64 hechos. Cada barra conserva su denominador.',
        'value_title': 'A · Efecto sobre Value', 'value_axis': 'ΔValue · nats / hecho',
        'general_title': 'B · Perturbación original', 'general_axis': 'g · nats / token válido',
        'value_positions': 'C · ΔValue por posición', 'general_positions': 'D · g por posición',
        'vetoed_unit': 'Unidad vetada',
        'effects_denominators': 'ΔValue: media igual por hecho (19); g: NLL original ponderada por tokens válidos.',
        'effects_scales': 'Mapas con igual peso. Los paneles usan escalas distintas; I / S8 / G4 según figura de competencia.',
        'effects_repeats': 'Dos baterías idénticas: se muestra una. No son dos réplicas independientes; sin barras de error.',
        'leader': ' · líder', 'matching_axis': 'g agregado · nats / token original válido',
        'matching_title': 'A · Caliper g: banda admisible [{low}, {high}]',
        'yes': 'Sí', 'no': 'No', 'calipers_title': 'B · Calipers originales',
        'calipers_axis': 'Cumple cada criterio',
        'position_difference_title': 'C · g por mapa, diferencia a U{target}',
        'position_difference_axis': 'Desglose descriptivo: nats / token',
        'matching_scale': 's_g = {scale}. La banda es un umbral de diseño; no un intervalo de confianza.',
        'matching_admission': 'La admisión requiere todos los calipers. El desglose por mapa no sustituye el criterio agregado.',
        'matching_result': 'Las dos baterías producen el mismo matching; no hay control admisible bajo V1.',
        'facts_title': 'Heterogeneidad de ΔValue por hecho',
        'facts_membership': 'Cada celda: media de dos paráfrasis × tres posiciones. Se conservan los 19 hechos common.',
        'facts_note': 'Descriptivo, sin IC ni p-valores. E00014 = Entity_00014. Dos baterías idénticas.',
        'prospective_title': 'V6 · PROSPECTIVO / NO EJECUTADO',
        'family': 'Familia', 'first_half': 'Steps [0,2000)', 'second_half': 'Steps [2000,4000)',
        'empty_sham': 'Sham vacío', 'fixed_unit': 'Unidad fija u',
        'distributed': 'Distribuido: π_b[(k+s) mod 4]',
        'same_schedule': 'Mismo calendario distribuido',
        'persistent_prefix': 'Prefijo idéntico a P_u', 'distributed_prefix': 'Prefijo idéntico a D_s',
        'first_calendar': 'Calendario fijado: steps 0–15', 'rescue_calendar': 'Frontera: steps 1992–2007',
        'optimizer_index': 'Índice global del optimizer', 'sham_legend': 'Sham / sin veto',
        'calendar_cells': 'La celda indica la unidad vetada, no la magnitud eliminada. Inferencia primaria: sin veto en todos.',
        'calendar_dose': 'Un sitio por forward; balance de oportunidades por step y bloque. No iguala COMMITs, RMS ni g.',
    },
    'en': {
        'competence_common': 'A · Common', 'competence_overall': 'B · Overall',
        'competence_rare': 'C · Rare', 'competence_unexposed': 'D · Unexposed',
        'competence_title': '{label}\n{facts} facts · {queries} questions',
        'gate': '80% gate', 'accuracy': 'Strict accuracy (%)',
        'positions_note': 'I: identity   ·   S8: shift +8   ·   G4: gaps 4',
        'competence_note': 'The gate applies to common facts; overall includes all 64 facts. Each bar retains its denominator.',
        'value_title': 'A · Effect on Value', 'value_axis': 'ΔValue · nats / fact',
        'general_title': 'B · Original-text disruption', 'general_axis': 'g · nats / valid token',
        'value_positions': 'C · ΔValue by position', 'general_positions': 'D · g by position',
        'vetoed_unit': 'Vetoed unit',
        'effects_denominators': 'ΔValue: equal weighting of 19 facts; g: original NLL weighted by valid tokens.',
        'effects_scales': 'Maps have equal weight. Panels use different scales; I / S8 / G4 as in the competence figure.',
        'effects_repeats': 'Two identical batteries: one shown. They are not independent replicates; no error bars.',
        'leader': ' · leader', 'matching_axis': 'Aggregate g · nats / original valid token',
        'matching_title': 'A · g caliper: admissible band [{low}, {high}]',
        'yes': 'Yes', 'no': 'No', 'calipers_title': 'B · Original calipers',
        'calipers_axis': 'Meets each criterion',
        'position_difference_title': 'C · g by map, difference from U{target}',
        'position_difference_axis': 'Descriptive breakdown: nats / token',
        'matching_scale': 's_g = {scale}. The band is a design threshold, not a confidence interval.',
        'matching_admission': 'Admission requires every caliper. The map breakdown does not replace the aggregate criterion.',
        'matching_result': 'Both batteries produce the same matching result; no admissible control under V1.',
        'facts_title': 'Heterogeneity of ΔValue across facts',
        'facts_membership': 'Each cell: mean of two paraphrases × three positions. All 19 common facts are retained.',
        'facts_note': 'Descriptive; no CIs or p-values. E00014 = Entity_00014. Two identical batteries.',
        'prospective_title': 'V6 · PROSPECTIVE / NOT EXECUTED',
        'family': 'Family', 'first_half': 'Steps [0,2000)', 'second_half': 'Steps [2000,4000)',
        'empty_sham': 'Empty sham', 'fixed_unit': 'Fixed unit u',
        'distributed': 'Distributed: π_b[(k+s) mod 4]',
        'same_schedule': 'Same distributed schedule',
        'persistent_prefix': 'Same prefix as P_u', 'distributed_prefix': 'Same prefix as D_s',
        'first_calendar': 'Fixed schedule: steps 0–15', 'rescue_calendar': 'Boundary: steps 1992–2007',
        'optimizer_index': 'Global optimizer index', 'sham_legend': 'Sham / no veto',
        'calendar_cells': 'The cell shows the vetoed unit, not the removed magnitude. Primary inference: no veto for all.',
        'calendar_dose': 'One site per forward; opportunity balance by step and block. COMMITs, RMS and g are not matched.',
    },
}


def require(value, message):
    if not value:
        raise ValueError(message)


def label(language, key, **values):
    require(language in LABELS, 'explicit es or en figure language required')
    return LABELS[language][key].format(**values)


def number(value, digits=3, signed=False, *, language='es'):
    require(language in LABELS, 'explicit es or en number format required')
    if round(float(value), digits) == 0:
        value = 0.
    text = f'{float(value):+.{digits}f}' if signed else f'{float(value):.{digits}f}'
    text = text.replace('-', '−')
    return text.replace('.', ',') if language == 'es' else text


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5,
        'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.titlepad': 9,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'text.color': NAVY, 'axes.labelcolor': NAVY, 'axes.edgecolor': GREY,
        'xtick.color': GREY, 'ytick.color': GREY, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6, 'grid.color': LIGHT,
        'grid.linewidth': .5, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'mathtext.fontset': 'dejavusans', 'figure.dpi': 120,
    })


def finish(fig, name, output, dpi):
    fig.savefig(output/(name+'.pdf'), metadata={'Creator': 'make_figures.py / matplotlib',
        'CreationDate': None, 'ModDate': None})
    fig.savefig(output/(name+'.png'), dpi=dpi)
    plt.close(fig)


def foot(fig, text, y=.025):
    fig.text(.04, y, text, ha='left', va='bottom', fontsize=7.1, color=GREY)


def load_units(data):
    batteries = data['measurement']['batteries']
    require(len(batteries) == 2, 'exactly two declared batteries required')
    ordered = [sorted(b['units'], key=lambda row: row['unit']) for b in batteries]
    require(all([row['unit'] for row in units] == [0, 1, 2, 3] for units in ordered), 'four complete units required')
    for left, right in zip(*ordered):
        for key in ('delta_value_nats', 'g', 'm', 'r', 'rho', 'p'):
            require(left[key] == right[key], 'displayed battery values differ; explicit alternate plotting is required')
        require(left['fact_effects'] == right['fact_effects'] and left['positions'] == right['positions'],
                'per-fact or per-position battery values differ')
    return ordered[0]


def positions_matrix(units, key):
    matrix = np.array([[next(row for row in unit['positions'] if row['position'] == position)[key]
                        for unit in units] for position in POSITIONS], dtype=float)
    require(matrix.shape == (3, 4) and np.isfinite(matrix).all(), 'finite complete position matrix required')
    return matrix


def vector_grid(ax, matrix, *, cmap, norm):
    rows, columns = matrix.shape
    artist = ax.pcolormesh(np.arange(columns+1)-.5, np.arange(rows+1)-.5, matrix,
        cmap=cmap, norm=norm, shading='flat', rasterized=False, edgecolors='none')
    ax.set_xlim(-.5, columns-.5); ax.set_ylim(rows-.5, -.5)
    return artist


def heatmap(ax, matrix, row_labels, column_labels, *, digits=3, signed=False, fontsize=7.5, extent=None, language='es'):
    maximum = max(float(np.max(np.abs(matrix))), 1e-9) if extent is None else extent
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0., vmax=maximum)
    image = vector_grid(ax, matrix, cmap=DIVERGING, norm=norm)
    ax.set_xticks(range(len(column_labels)), column_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            red, green, blue, _ = DIVERGING(norm(value))
            lightness = .2126*red+.7152*green+.0722*blue
            ax.text(column, row, number(value, digits, signed, language=language), ha='center', va='center',
                    color='white' if lightness < .59 else NAVY, fontsize=fontsize)
    return image


def competence(data, output, dpi, *, language='es'):
    rows = {(row['position'], row['stratum']): row for row in data['reference']['competence']}
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.65), sharey=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.16, top=.865, hspace=.65, wspace=.19)
    panels = tuple((stratum, label(language, 'competence_'+stratum)) for stratum in ('common', 'overall', 'rare', 'unexposed'))
    for ax, (stratum, panel_label) in zip(axes.flat, panels):
        values = [rows[position, stratum] for position in POSITIONS]
        require(len({(row['facts'], row['queries']) for row in values}) == 1, 'position denominators must be explicit and consistent')
        percentages = [100*row['strict_accuracy'] for row in values]
        require(all(abs(row['strict_accuracy']-row['correct']/row['queries']) < 1e-12 for row in values),
                'strict accuracy must match the displayed full denominator')
        ax.bar(range(3), percentages, color=POSITION_COLORS, width=.58, zorder=3)
        for index, (row, percentage) in enumerate(zip(values, percentages)):
            ax.text(index, percentage+3, f"{number(percentage, 1, language=language)}%\n{row['correct']}/{row['queries']}",
                    ha='center', va='bottom', fontsize=7.8, linespacing=1.15)
        ax.set_title(label(language, 'competence_title', label=panel_label, facts=values[0]['facts'], queries=values[0]['queries']), loc='left', fontsize=9)
        ax.set_xticks(range(3), POSITION_SHORT)
        ax.set_ylim(0, 121); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_xlim(-.6, 2.6)
        ax.yaxis.grid(True, zorder=0)
        if stratum == 'common':
            ax.axhline(80, color=GREY, linestyle=(0, (3, 2)), linewidth=.9, zorder=2)
            ax.text(2.53, 78, label(language, 'gate'), ha='right', va='top', fontsize=7, color=GREY)
    axes[0, 0].set_ylabel(label(language, 'accuracy'))
    axes[1, 0].set_ylabel(label(language, 'accuracy'))
    foot(fig, label(language, 'positions_note'), y=.078)
    foot(fig, label(language, 'competence_note'), y=.025)
    finish(fig, 'competence', output, dpi)


def channel_effects(data, output, dpi, *, language='es'):
    units = load_units(data)
    delta = np.array([row['delta_value_nats'] for row in units])
    general = np.array([row['g'] for row in units])
    fig = plt.figure(figsize=(WIDTH, 4.8))
    grid = fig.add_gridspec(2, 2, left=.12, right=.97, bottom=.265, top=.92,
                           wspace=.38, hspace=.7, height_ratios=[1.15, .9])
    for column, (values, title, ylabel) in enumerate(((delta, label(language, 'value_title'), label(language, 'value_axis')),
                                                     (general, label(language, 'general_title'), label(language, 'general_axis')))):
        ax = fig.add_subplot(grid[0, column])
        ax.bar(range(4), values, color=[TEAL, NAVY, NAVY, NAVY], width=.6, zorder=3)
        ax.axhline(0, color=GREY, linewidth=.7)
        span = max(float(np.ptp(values)), float(np.max(abs(values))), .02)
        for i, value in enumerate(values):
            ax.text(i, value+(.035*span if value >= 0 else -.04*span), number(value, 3, language=language),
                    ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
        ax.set_ylim(min(0, float(values.min()))-.18*span, float(values.max())+.2*span)
        ax.set_xticks(range(4), [f'U{u}' for u in range(4)])
        ax.set_title(title, loc='left'); ax.set_ylabel(ylabel); ax.yaxis.grid(True, zorder=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: number(value, 2, language=language)))
    for column, (key, title) in enumerate((('delta_value_nats', label(language, 'value_positions')), ('g', label(language, 'general_positions')))):
        ax = fig.add_subplot(grid[1, column])
        matrix = positions_matrix(units, key)
        heatmap(ax, matrix, POSITION_SHORT, [f'U{u}' for u in range(4)], digits=3, fontsize=7.3, language=language)
        ax.set_title(title, loc='left'); ax.set_xlabel(label(language, 'vetoed_unit'))
    foot(fig, label(language, 'effects_denominators'), y=.115)
    foot(fig, label(language, 'effects_scales'), y=.068)
    foot(fig, label(language, 'effects_repeats'), y=.022)
    finish(fig, 'channel-effects', output, dpi)


def matching(data, output, dpi, *, language='es'):
    units = load_units(data)
    records = data['measurement']['matching']
    require(len(records) == 2 and records[0]['target_unit'] == records[1]['target_unit'], 'consistent declared matching target required')
    record = records[0]; target = record['target_unit']
    candidates = sorted(record['candidates'], key=lambda row: row['unit'])
    target_g = units[target]['g']; scale = record['general_loss_scale']
    low, high = target_g-scale, target_g+scale
    fig = plt.figure(figsize=(WIDTH, 4.8))
    grid = fig.add_gridspec(2, 2, left=.12, right=.97, bottom=.265, top=.91,
        wspace=.42, hspace=.76, height_ratios=[1.1, 1.])
    ax = fig.add_subplot(grid[0, :])
    ax.axvspan(low, high, color=AMBER, alpha=.2, linewidth=0)
    ax.axvline(low, color=AMBER, linewidth=1)
    ax.axvline(high, color=AMBER, linewidth=1)
    ax.axvline(target_g, color=TEAL, linestyle=(0, (3, 2)), linewidth=1)
    passing = {row['unit']: row['calipers']['general_nll'] for row in candidates}
    for unit in units:
        index, value = unit['unit'], unit['g']
        marker = 'D' if index == target else ('o' if passing[index] else 'x')
        color = TEAL if index == target else NAVY
        ax.scatter(value, index, marker=marker, color=color, s=42, linewidths=1.5, zorder=4)
        ax.annotate(number(value, 4, language=language), (value, index), xytext=(8, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=8.3, bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none', 'pad': .5})
    ax.set_xlim(min(0, min(unit['g'] for unit in units))-.005, max(high, max(unit['g'] for unit in units))+.033)
    ax.set_ylim(3.65, -.55)
    ax.set_yticks(range(4), [f'U{u}'+(label(language, 'leader') if u == target else '') for u in range(4)])
    ax.set_xlabel(label(language, 'matching_axis'))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: number(value, 3, language=language)))
    ax.xaxis.grid(True, zorder=0)
    ax.set_title(label(language, 'matching_title', low=number(low, 4, language=language), high=number(high, 4, language=language)), loc='left')
    caliper_names = ('general_nll', 'update_rms', 'relative_update_rms', 'admission_fraction')
    matrix = np.array([[int(row['calipers'][name]) for name in caliper_names] for row in candidates])
    ax = fig.add_subplot(grid[1, 0])
    vector_grid(ax, matrix, cmap=ListedColormap(['#FFF0D0', '#DEEFEB']), norm=BoundaryNorm([-.5, .5, 1.5], 2))
    ax.set_xticks(range(4), ['g', 'm', 'm/r', 'p'])
    ax.set_yticks(range(len(candidates)), [f"U{row['unit']}" for row in candidates])
    ax.tick_params(length=0, pad=6)
    for spine in ax.spines.values(): spine.set_visible(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, label(language, 'yes' if matrix[row, column] else 'no'), ha='center', va='center', fontsize=9, weight='bold')
    ax.set_title(label(language, 'calipers_title'), loc='left'); ax.set_xlabel(label(language, 'calipers_axis'))
    ax = fig.add_subplot(grid[1, 1])
    g = positions_matrix(units, 'g')
    differences = np.column_stack([g[:, row['unit']]-g[:, target] for row in candidates])
    heatmap(ax, differences, POSITION_SHORT, [f"U{row['unit']}" for row in candidates], signed=True, digits=3, fontsize=7.4, language=language)
    ax.set_title(label(language, 'position_difference_title', target=target), loc='left', fontsize=8.8)
    ax.set_xlabel(label(language, 'position_difference_axis'))
    foot(fig, label(language, 'matching_scale', scale=number(scale, 6, language=language)), y=.115)
    foot(fig, label(language, 'matching_admission'), y=.068)
    foot(fig, label(language, 'matching_result'), y=.022)
    finish(fig, 'matching', output, dpi)


def fact_heterogeneity(data, output, dpi, *, language='es'):
    units = load_units(data)
    facts = [row['fact_id'] for row in units[0]['fact_effects']]
    require(len(facts) == len(set(facts)) == 19, 'all19 common facts required')
    matrix = np.column_stack([[next(row['delta_value_nats'] for row in unit['fact_effects'] if row['fact_id'] == fact)
                              for fact in facts] for unit in units])
    require(np.isfinite(matrix).all(), 'finite fact-effect matrix required')
    labels = [fact.replace('Entity_', 'E') for fact in facts]
    fig = plt.figure(figsize=(WIDTH, 5.65))
    ax = fig.add_axes([.25, .21, .6, .70])
    image = heatmap(ax, matrix, labels, [f'U{u}' for u in range(4)], digits=3, fontsize=7.7, language=language)
    ax.set_title(label(language, 'facts_title'), loc='left', fontsize=10)
    ax.set_xlabel(label(language, 'vetoed_unit'))
    cax = fig.add_axes([.88, .2, .025, .6])
    bar = fig.colorbar(image, cax=cax)
    bar.solids.set_rasterized(False)
    # Overlap adjacent vector colorbar patches to avoid PDF antialias seams.
    bar.solids.set_edgecolor('face')
    bar.set_label('ΔValue · nats', fontsize=8)
    foot(fig, label(language, 'facts_membership'), y=.073)
    foot(fig, label(language, 'facts_note'), y=.025)
    finish(fig, 'fact-heterogeneity', output, dpi)


def prospective_design(data, output, dpi, *, language='es'):
    design = data['design_v6']; arms = design['arms']
    require(len(arms) == 17 and arms[0] == 'H' and design['horizon'] == 4000 and design['rescue_step'] == 2000,
            'complete prospective17-arm A4000/rescue2000 design required')
    windows = {row['label']: row for row in design['calendar_examples']}
    fig = plt.figure(figsize=(WIDTH, 132/25.4))
    fig.text(.045, .98, label(language, 'prospective_title'), ha='left', va='top', color=TEAL, weight='bold', fontsize=11)
    families = fig.add_axes([.04, .755, .92, .175]); families.axis('off')
    content = [
        ['H', label(language, 'empty_sham'), label(language, 'empty_sham')],
        ['P0–P3', label(language, 'fixed_unit'), label(language, 'fixed_unit')],
        ['D0–D3', label(language, 'distributed'), label(language, 'same_schedule')],
        ['RP0–RP3', label(language, 'persistent_prefix'), label(language, 'empty_sham')],
        ['RD0–RD3', label(language, 'distributed_prefix'), label(language, 'empty_sham')],
    ]
    table = families.table(cellText=content, colLabels=[label(language, 'family'), label(language, 'first_half'), label(language, 'second_half')],
                           colWidths=[.18, .42, .40], bbox=[0, 0, 1, 1], cellLoc='left', colLoc='left')
    table.auto_set_font_size(False); table.set_fontsize(7.8)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(.4); cell.set_edgecolor(LIGHT)
        if row == 0:
            cell.set_facecolor('#EDF2F4'); cell.set_text_props(weight='bold', color=NAVY)
        else:
            cell.set_facecolor('white'); cell.set_text_props(color=NAVY)
    cmap = ListedColormap(['#EFF2F4', NAVY, TEAL, AMBER, SLATE])
    norm = BoundaryNorm(np.arange(-.5, 5.5), cmap.N)
    for column, key in enumerate(('first_16', 'rescue_boundary')):
        window = windows[key]
        steps = window['steps']; rows = {item['arm']: item['veto_units'] for item in window['arms']}
        require(set(rows) == set(arms) and len(steps) == 16 and all(len(rows[arm]) == 16 for arm in arms), 'complete real16-step calendar window required')
        values = np.array([[0 if unit is None else unit+1 for unit in rows[arm]] for arm in arms], dtype=int)
        ax = fig.add_axes([.10+column*.46, .23, .405, .45])
        vector_grid(ax, values, cmap=cmap, norm=norm)
        ax.set_yticks(range(17), arms if column == 0 else ['']*17)
        ax.tick_params(length=0, pad=4)
        labels = [str(step) if index % 4 == 0 or step == 2000 else '' for index, step in enumerate(steps)]
        ax.set_xticks(range(16), labels)
        ax.set_xticks(np.arange(-.5, 16, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 17, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=.6); ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        for row in range(17):
            for col in range(16):
                if values[row, col]:
                    ax.text(col, row, str(values[row, col]-1), ha='center', va='center', fontsize=5.8,
                            color='white' if values[row, col] in (1, 2) else NAVY)
        for boundary in (0.5, 4.5, 8.5, 12.5):
            ax.axhline(boundary, color=GREY, linewidth=.8)
        if 2000 in steps:
            ax.axvline(steps.index(2000)-.5, color=TEAL, linewidth=1.7)
        ax.set_title(label(language, 'first_calendar' if column == 0 else 'rescue_calendar'), fontsize=8.5, loc='left', pad=8)
        ax.set_xlabel(label(language, 'optimizer_index'), fontsize=7.7)
    handles = [Patch(facecolor=cmap(0), edgecolor=LIGHT, label=label(language, 'sham_legend'))]
    handles += [Patch(facecolor=cmap(unit+1), label=f'U{unit}') for unit in range(4)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .118), ncols=5,
               frameon=False, fontsize=7.5, handlelength=1.4, columnspacing=1.3)
    foot(fig, label(language, 'calendar_cells'), y=.058)
    foot(fig, label(language, 'calendar_dose'), y=.016)
    finish(fig, 'prospective-design', output, dpi)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent/'data/report-data.json')
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--language', choices=tuple(LABELS), default='es')
    parser.add_argument('--dpi', type=int, default=220)
    args = parser.parse_args(argv)
    require(100 <= args.dpi <= 600, 'PNG preview DPI must be100..600')
    data = json.loads(args.data.read_text(encoding='utf-8'))
    require(data['schema_version'] == 1, 'report-data schema_version1 required')
    output = args.outdir
    if output is None:
        output = Path(__file__).resolve().parent/'figures'
        if args.language == 'en':
            output = output/'en'
    output.mkdir(parents=True, exist_ok=True)
    style()
    for render in (competence, channel_effects, matching, fact_heterogeneity, prospective_design):
        render(data, output, args.dpi, language=args.language)
    print(f'Rendered five vector PDFs and five PNG previews ({args.language}) from report-data.json.')


if __name__ == '__main__':
    main()
