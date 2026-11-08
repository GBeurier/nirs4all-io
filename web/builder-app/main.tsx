/*
 * Host mount for the accepted nirs4all-ui `.1` column-mapping components.
 *
 * Composes three public components of `nirs4all-ui/datasetBuilder` (no wizard container, no
 * auto-detection, no validation or export helpers) into one React root per call:
 *   SourceSummaryCard, RoleSelectionCards, ColumnMappingTable.
 * The handle contract matches the shared browser entry: `update` replaces the complete props,
 * `unmount` releases the root. The host (dataset-builder.js) owns every id, role and selection.
 *
 * Bundled by build-dataset-builder.mjs from the verified `.1` package into
 * vendor/ui-columns/column-mapping.mjs. `n4ui-datasetbuilder/*` is an esbuild alias for the
 * extracted package's dist/datasetBuilder directory.
 */
import { createElement } from "react";
import { flushSync } from "react-dom";
import { createRoot } from "react-dom/client";

import { ColumnMappingTable } from "n4ui-datasetbuilder/ColumnMappingTable.js";
import { RoleSelectionCards } from "n4ui-datasetbuilder/RoleSelectionCards.js";
import { SourceSummaryCard } from "n4ui-datasetbuilder/SourceSummaryCard.js";

type Props = Record<string, any>;

function ColumnEditor(p: Props) {
  return createElement(
    "div",
    { className: "dsb dsb-host" },
    createElement(SourceSummaryCard, { source: p.source, sources: p.sources, onChangeSource: p.onChangeSource, locale: "en" }),
    createElement(RoleSelectionCards, { roles: p.roles, selectedCount: p.selectedCount, onAssignRole: p.onAssignRole, locale: "en" }),
    createElement(ColumnMappingTable, {
      columns: p.source.columns,
      onToggleColumn: p.onToggleColumn,
      onToggleAll: p.onToggleAll,
      onAssignColumnRole: p.onAssignColumnRole,
      filter: p.filter,
      onFilterChange: p.onFilterChange,
      showTip: false,
      locale: "en",
    }),
  );
}

export function mountColumnMapping(host: HTMLElement, props: Props) {
  const root = createRoot(host);
  let active = true;
  const render = (next: Props) => flushSync(() => root.render(createElement(ColumnEditor, next)));
  render(props);
  return {
    update(next: Props) {
      if (!active) throw new Error("column mapping: update() called after unmount()");
      render(next);
    },
    unmount() {
      if (!active) return;
      active = false;
      root.unmount();
    },
  };
}
