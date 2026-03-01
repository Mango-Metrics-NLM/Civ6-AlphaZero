/**
 * MCTS Tree Visualization using D3.js force-directed graph.
 */
class MCTSTreeViz {
    constructor(containerId, width = 300, height = 200) {
        this.containerId = containerId;
        this.width = width;
        this.height = height;
        this.svg = null;
    }

    render(treeData) {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        container.innerHTML = '';

        if (!treeData || !treeData.nodes || treeData.nodes.length === 0) {
            container.innerHTML = '<p class="muted">No tree data</p>';
            return;
        }

        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);

        const nodes = treeData.nodes;
        const edges = treeData.edges;

        // Scale visits to node radius
        const maxVisits = Math.max(1, ...nodes.map(n => n.visits));
        const radiusScale = d3.scaleSqrt().domain([0, maxVisits]).range([4, 16]);

        // Color by value: green=positive, red=negative
        const colorScale = d3.scaleLinear()
            .domain([-1, 0, 1])
            .range(['#e74c3c', '#8e8e9e', '#2ecc71']);

        // D3 force simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(edges).id(d => d.id).distance(40))
            .force('charge', d3.forceManyBody().strength(-50))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collide', d3.forceCollide().radius(d => radiusScale(d.visits) + 2));

        // Edges
        const link = this.svg.selectAll('.tree-link')
            .data(edges)
            .enter()
            .append('line')
            .attr('class', 'tree-link')
            .attr('stroke', '#2a2a4a')
            .attr('stroke-width', 1);

        // Edge labels
        const linkLabel = this.svg.selectAll('.tree-link-label')
            .data(edges)
            .enter()
            .append('text')
            .attr('class', 'tree-link-label')
            .attr('fill', '#8e8e9e')
            .attr('font-size', '8px')
            .attr('text-anchor', 'middle')
            .text(d => d.action);

        // Nodes
        const node = this.svg.selectAll('.tree-node')
            .data(nodes)
            .enter()
            .append('circle')
            .attr('class', 'tree-node')
            .attr('r', d => radiusScale(d.visits))
            .attr('fill', d => colorScale(d.value))
            .attr('stroke', '#eaeaea')
            .attr('stroke-width', 0.5);

        // Tooltips
        node.append('title')
            .text(d => `Visits: ${d.visits}\nValue: ${d.value}\nPrior: ${d.prior}`);

        simulation.on('tick', () => {
            link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
            linkLabel.attr('x', d => (d.source.x + d.target.x) / 2)
                     .attr('y', d => (d.source.y + d.target.y) / 2);
            node.attr('cx', d => d.x).attr('cy', d => d.y);
        });

        // Stop after settling
        setTimeout(() => simulation.stop(), 3000);
    }

    async fetchAndRender(sessionId) {
        try {
            const resp = await fetch(`/api/mcts/tree?session_id=${sessionId}`);
            const data = await resp.json();
            this.render(data);
        } catch (err) {
            console.error('Failed to fetch MCTS tree:', err);
        }
    }
}

// Auto-init for play page
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('mcts-tree-container');
    if (container) {
        window.mctsViz = new MCTSTreeViz('mcts-tree-container', 280, 200);
    }

    document.getElementById('btn-view-tree')?.addEventListener('click', () => {
        if (window.atariGame?.sessionId && window.mctsViz) {
            window.mctsViz.fetchAndRender(window.atariGame.sessionId);
        }
    });
});
