import { useEffect, useRef, useState } from 'react'
import { Download, Maximize, Minus, Plus } from 'lucide-react'
import { predicateLabel, typeColor, typeLabel } from '../../lib/graphColors'
import { THEME_EVENT, cssColor } from '../../lib/theme'
import { Spinner } from '../ui'

/**
 * Interactive knowledge-graph canvas, drawn with Cytoscape.js.
 *
 * Cytoscape is imported dynamically, so its ~400 KB only loads when the Graph
 * tab is opened. The instance is created once and then updated by diffing:
 * expanding a node adds its neighbours next to it rather than rebuilding and
 * re-laying out the whole graph, so what the user was looking at stays put.
 *
 * `graph.key` marks a fresh graph (overview, or focus on a new entity): a new key
 * clears the canvas and lays it out from scratch.
 */

const CANVAS_HEIGHT = 540

/**
 * Cytoscape paints to a canvas, so it cannot inherit a CSS variable the way the
 * rest of the app does -- every colour has to be resolved to a literal first.
 * Built as a function rather than a constant so the palette can be re-read when
 * the theme changes; `cy.style()` then swaps it without touching the layout, so
 * whatever the user was looking at stays exactly where it was.
 */
function buildStyle() {
  const canvas = cssColor('canvas', '#0b0f19')
  const ink = cssColor('ink', '#e8edf7')
  const inkMuted = cssColor('ink-muted', '#94a1bd')
  const line = cssColor('line', '#2a3349')
  const brandSoft = cssColor('brand-soft', '#818cf8')
  const warn = cssColor('warn', '#fbbf24')

  return [
    {
      selector: 'node',
      style: {
        'background-color': 'data(color)',
        'border-width': 1.5,
        // The page colour, so nodes read as cut out of the background.
        'border-color': canvas,
        width: 'mapData(degree, 0, 40, 16, 46)',
        height: 'mapData(degree, 0, 40, 16, 46)',
        label: 'data(name)',
        color: ink,
        'font-size': 10,
        'font-family': 'Inter, ui-sans-serif, system-ui, sans-serif',
        'text-valign': 'bottom',
        'text-margin-y': 4,
        'text-wrap': 'ellipsis',
        'text-max-width': '120px',
        'text-outline-color': canvas,
        'text-outline-width': 2,
        // Labels disappear when zoomed out far enough to be unreadable anyway.
        'min-zoomed-font-size': 6,
        'overlay-opacity': 0,
        'transition-property': 'opacity',
        'transition-duration': '150ms',
      },
    },
    {
      selector: 'node.center',
      style: { 'border-width': 3, 'border-color': ink },
    },
    {
      selector: 'node.picked',
      style: {
        'border-width': 3,
        'border-color': warn,
        'font-weight': 600,
        'z-index': 10,
      },
    },
    {
      selector: 'edge',
      style: {
        width: 'mapData(weight, 1, 6, 1, 4)',
        'line-color': line,
        'target-arrow-color': line,
        'target-arrow-shape': 'triangle',
        'arrow-scale': 0.8,
        'curve-style': 'bezier',
        label: 'data(label)',
        color: inkMuted,
        'font-size': 8,
        'text-rotation': 'autorotate',
        'text-background-color': canvas,
        'text-background-opacity': 0.85,
        'text-background-padding': '1px',
        'min-zoomed-font-size': 7,
        'overlay-opacity': 0,
      },
    },
    {
      selector: 'edge.lit',
      style: {
        'line-color': brandSoft,
        'target-arrow-color': brandSoft,
        color: brandSoft,
        'z-index': 9,
      },
    },
    { selector: '.faded', style: { opacity: 0.12 } },
  ]
}

/**
 * Fit the graph into view, recovering a viewport left in an impossible state.
 * Fitting while the container measures zero can produce a non-finite zoom, and
 * a canvas with a NaN transform draws nothing at all -- which looks exactly like
 * "the graph never loaded", even though a PNG export of it comes out fine.
 */
function fitView(cy) {
  if (!cy || cy.destroyed() || cy.nodes().length === 0) return
  if (cy.width() === 0 || cy.height() === 0) return

  cy.fit(undefined, 30)

  const zoom = cy.zoom()
  const pan = cy.pan()
  if (!Number.isFinite(zoom) || zoom <= 0 || !Number.isFinite(pan.x) || !Number.isFinite(pan.y)) {
    cy.zoom(1)
    cy.center()
  }

  // Draw now rather than waiting for the next event. Cytoscape suppresses
  // on-screen redraws while a batch is open, and an exception inside one can
  // leave it open forever -- a state where the live canvas stays blank while
  // PNG export, which renders to its own canvas, still works.
  cy.forceRender?.()
}

/**
 * Cytoscape draws into canvases nested in a wrapper div that it sizes from its
 * own measurement of the container, with `overflow: hidden`. If that
 * measurement was taken while the container had no size, the wrapper stays
 * small and clips every canvas away -- a blank graph that still exports to PNG
 * perfectly, because export renders to a canvas of its own. Re-assert the size.
 */
function ensureLayerSize(container, cy) {
  const inner = container?.firstElementChild
  if (!inner || !container.clientWidth || !container.clientHeight) return false

  const tooNarrow = inner.clientWidth < container.clientWidth - 2
  const tooShort = inner.clientHeight < container.clientHeight - 2
  if (!tooNarrow && !tooShort) return false

  inner.style.width = `${container.clientWidth}px`
  inner.style.height = `${container.clientHeight}px`
  cy?.resize()
  return true
}

function layoutOptions(name, { fresh, animate }) {
  if (name === 'rings') {
    return {
      name: 'concentric',
      // Depth 0 (the focused entity) in the middle, each hop one ring further out.
      concentric: (node) => 10 - (node.data('depth') ?? 3),
      levelWidth: () => 1,
      minNodeSpacing: 18,
      padding: 30,
      animate,
      animationDuration: 350,
    }
  }
  return {
    name: 'cose',
    randomize: fresh,
    animate,
    animationDuration: 400,
    nodeRepulsion: () => 9000,
    idealEdgeLength: () => 90,
    edgeElasticity: () => 80,
    gravity: 0.35,
    numIter: 1200,
    padding: 30,
  }
}

function toNode(node, position) {
  return {
    group: 'nodes',
    data: {
      id: node.id,
      name: node.name,
      type: node.type,
      color: typeColor(node.type),
      // mapData extrapolates past its range, so the size inputs are clamped.
      degree: Math.min(node.degree ?? 0, 40),
      mentions: node.mentions ?? 0,
      depth: node.depth,
    },
    ...(position ? { position } : {}),
  }
}

function toEdge(edge) {
  return {
    group: 'edges',
    data: {
      id: `e:${edge.id}`,
      source: edge.source,
      target: edge.target,
      label: predicateLabel(edge.predicate),
      weight: Math.min(edge.weight ?? 1, 6),
    },
  }
}

export function GraphCanvas({ graph, centerId, selectedId, layout = 'force', onSelect, onExpand, onClear }) {
  const containerRef = useRef(null)
  const cyRef = useRef(null)
  const layoutRef = useRef(null)
  const keyRef = useRef(null)
  const [status, setStatus] = useState('loading')
  // Bumped whenever a new Cytoscape instance exists. React StrictMode mounts
  // effects twice in development, so the instance the first render created is
  // destroyed and replaced; without this the element-sync effect would not
  // re-run for the replacement and the canvas would stay empty.
  const [instance, setInstance] = useState(0)

  // Latest callbacks, so the Cytoscape listeners bound once never go stale.
  const handlers = useRef({})
  handlers.current = { onSelect, onExpand, onClear }

  /* ---- create the instance once ------------------------------------------ */
  useEffect(() => {
    let cancelled = false
    let observer = null

    import('cytoscape')
      .then(({ default: cytoscape }) => {
        if (cancelled || !containerRef.current) return
        const cy = cytoscape({
          container: containerRef.current,
          style: buildStyle(),
          minZoom: 0.15,
          maxZoom: 3,
          wheelSensitivity: 0.25,
          boxSelectionEnabled: false,
          autounselectify: true,
        })
        cy.on('tap', 'node', (event) => handlers.current.onSelect?.(event.target.id()))
        cy.on('dbltap', 'node', (event) => handlers.current.onExpand?.(event.target.id()))
        cy.on('tap', (event) => {
          if (event.target === cy) handlers.current.onClear?.()
        })
        cy.on('mouseover', 'node', () => (containerRef.current.style.cursor = 'pointer'))
        cy.on('mouseout', 'node', () => (containerRef.current.style.cursor = ''))

        // Cytoscape sizes its canvases from the container, and measuring it at
        // zero (a tab that was still laying out) leaves the viewport pointing
        // nowhere -- the graph is there but nothing is drawn. Re-measure on
        // every container resize, and refit once it first has real size.
        let hadSize = cy.width() > 0 && cy.height() > 0
        observer = new ResizeObserver(() => {
          cy.resize()
          ensureLayerSize(containerRef.current, cy)
          const hasSize = cy.width() > 0 && cy.height() > 0
          if (hasSize && !hadSize && cy.nodes().length) fitView(cy)
          hadSize = hasSize
        })
        observer.observe(containerRef.current)

        cyRef.current = cy
        // A new instance holds no elements, so the next sync must rebuild them.
        keyRef.current = null
        setStatus('ready')
        setInstance((value) => value + 1)

        // Handy in the browser console when a graph looks wrong:
        //   __sureCy.nodes().length, __sureCy.zoom(), __sureCy.width()
        if (import.meta.env.DEV) window.__sureCy = cy
      })
      .catch((err) => {
        console.error('Could not load cytoscape', err)
        if (!cancelled) setStatus('missing')
      })

    return () => {
      cancelled = true
      observer?.disconnect()
      layoutRef.current?.stop()
      cyRef.current?.destroy()
      cyRef.current = null
    }
  }, [])

  /* ---- repaint when the palette changes ----------------------------------- */
  useEffect(() => {
    const repaint = () => {
      // Swapping the stylesheet re-colours what is drawn without re-running the
      // layout, so the view the user had stays put through the switch.
      cyRef.current?.style(buildStyle())
    }
    window.addEventListener(THEME_EVENT, repaint)
    return () => window.removeEventListener(THEME_EVENT, repaint)
  }, [instance])

  function runLayout({ fresh }) {
    const cy = cyRef.current
    if (!cy || cy.nodes().length === 0) return
    layoutRef.current?.stop()
    // Lay out against the container's real size, not whatever it was when the
    // instance was created.
    cy.resize()
    const many = cy.nodes().length > 120
    const layoutRun = cy.layout(layoutOptions(layout, { fresh, animate: !fresh && !many }))
    layoutRef.current = layoutRun
    layoutRun.one('layoutstop', () => fitView(cy))
    layoutRun.run()
    // A layout that fits against a zero-sized viewport can leave a non-finite
    // zoom, which draws nothing at all; the frame after mount is sized properly.
    requestAnimationFrame(() => {
      if (cyRef.current === cy) {
        cy.resize()
        ensureLayerSize(containerRef.current, cy)
        fitView(cy)
      }
    })
  }

  /* ---- sync the elements with `graph` -------------------------------------- */
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return

    const fresh = keyRef.current !== graph.key
    keyRef.current = graph.key

    const nodeIds = new Set(graph.nodes.map((n) => n.id))
    // An edge to a node that is not drawn would make Cytoscape throw.
    const edges = graph.edges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
    const edgeIds = new Set(edges.map((e) => `e:${e.id}`))

    let changed = fresh
    cy.batch(() => {
      if (fresh) {
        cy.elements().remove()
      } else {
        const stale = cy.elements().filter((el) =>
          el.isNode() ? !nodeIds.has(el.id()) : !edgeIds.has(el.id())
        )
        if (stale.length) {
          stale.remove()
          changed = true
        }
      }

      // New nodes start around the node they were expanded from, so the layout
      // grows the graph outward from there instead of scattering it.
      const anchor = graph.anchor ? cy.getElementById(graph.anchor) : null
      const origin = anchor && anchor.nonempty() ? anchor.position() : null

      const additions = []
      for (const node of graph.nodes) {
        const existing = cy.getElementById(node.id)
        if (existing.nonempty()) {
          existing.data({ degree: Math.min(node.degree ?? 0, 40), depth: node.depth })
          continue
        }
        const position =
          !fresh && origin
            ? { x: origin.x + (Math.random() - 0.5) * 80, y: origin.y + (Math.random() - 0.5) * 80 }
            : undefined
        additions.push(toNode(node, position))
      }
      for (const edge of edges) {
        if (cy.getElementById(`e:${edge.id}`).empty()) additions.push(toEdge(edge))
      }
      if (additions.length) {
        cy.add(additions)
        changed = true
      }
    })

    if (changed) runLayout({ fresh })
    cy.forceRender?.()
    // runLayout reads `layout`; it is re-run separately when that changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph, instance])

  /* ---- switching layout re-arranges what is on screen ---------------------- */
  useEffect(() => {
    if (cyRef.current) runLayout({ fresh: false })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout])

  /* ---- highlight the focused and selected entities ------------------------ */
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.batch(() => {
      cy.elements().removeClass('faded lit picked center')
      if (centerId) cy.getElementById(centerId).addClass('center')
      if (!selectedId) return
      const node = cy.getElementById(selectedId)
      if (node.empty()) return
      const hood = node.closedNeighborhood()
      cy.elements().not(hood).addClass('faded')
      hood.edges().addClass('lit')
      node.addClass('picked')
    })
  }, [selectedId, centerId, graph, instance])

  function zoomBy(factor) {
    const cy = cyRef.current
    if (!cy) return
    cy.zoom({
      level: cy.zoom() * factor,
      renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 },
    })
  }

  function exportPng() {
    const cy = cyRef.current
    if (!cy) return
    const uri = cy.png({ full: true, scale: 2, bg: cssColor('canvas', '#0b0f19') })
    const link = document.createElement('a')
    link.href = uri
    link.download = 'knowledge-graph.png'
    link.click()
  }

  const presentTypes = [...new Set(graph.nodes.map((n) => n.type || 'OTHER'))].sort()

  // The size of the next two boxes is what Cytoscape measures itself against,
  // so it is set inline rather than through utility classes. A container that
  // computes to zero height draws nothing at all while still exporting a
  // perfect PNG, which is a baffling thing to debug.
  return (
    <div
      className="relative overflow-hidden rounded-lg border border-line bg-canvas"
      style={{ height: CANVAS_HEIGHT }}
    >
      <div ref={containerRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} />

      {status === 'loading' && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Spinner label="Loading graph renderer..." />
        </div>
      )}

      {status === 'missing' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-6 text-center">
          <p className="text-sm font-medium text-ink">The graph renderer is not installed</p>
          <p className="max-w-sm text-xs leading-relaxed text-ink-muted">
            Run <code className="rounded bg-surface-3 px-1">npm install</code> in the{' '}
            <code className="rounded bg-surface-3 px-1">frontend</code> folder, then restart{' '}
            <code className="rounded bg-surface-3 px-1">npm run dev</code>. The entity and relation
            lists work without it.
          </p>
        </div>
      )}

      {status === 'ready' && graph.nodes.length === 0 && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <p className="text-xs text-ink-faint">No connected entities to draw.</p>
        </div>
      )}

      {status === 'ready' && (
        <>
          <div className="absolute right-2 top-2 flex flex-col gap-1">
            {[
              [Plus, 'Zoom in', () => zoomBy(1.25)],
              [Minus, 'Zoom out', () => zoomBy(0.8)],
              [Maximize, 'Fit to screen', () => fitView(cyRef.current)],
              [Download, 'Download as PNG', exportPng],
            ].map(([Icon, label, action]) => (
              <button
                key={label}
                onClick={action}
                title={label}
                aria-label={label}
                className="flex h-7 w-7 items-center justify-center rounded-md border border-line bg-surface/90 text-ink-muted backdrop-blur-sm hover:text-ink"
              >
                <Icon className="h-3.5 w-3.5" />
              </button>
            ))}
          </div>

          {presentTypes.length > 0 && (
            <div className="pointer-events-none absolute bottom-2 left-2 flex max-w-[70%] flex-wrap gap-x-3 gap-y-1 rounded-md border border-line bg-surface/90 px-2.5 py-1.5 backdrop-blur-sm">
              {presentTypes.map((type) => (
                <span key={type} className="flex items-center gap-1.5 text-[10px] text-ink-muted">
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: typeColor(type) }} />
                  {typeLabel(type)}
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
