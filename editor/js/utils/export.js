/**
 * Copyright (c) 2024, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * Generates a flow configuration from the graph
 * @param {LGraph} graphInstance - The LiteGraph instance
 * @returns {FlowConfig} The generated flow configuration
 * @throws {Error} If the graph is invalid
 */
export function generateFlowConfig(graphInstance) {
  if (!graphInstance) {
    throw new Error("No graph instance provided");
  }

  if (!graphInstance._nodes || !Array.isArray(graphInstance._nodes)) {
    throw new Error("No nodes found in the graph");
  }

  const nodes = graphInstance._nodes;
  let startNode = nodes.find(
    (node) => node.constructor.name === "PipecatStartNode",
  );

  if (!startNode) {
    throw new Error("No start node found in the flow");
  }

  /**
   * Adds handler token if needed
   * @param {Object} functionConfig - Function configuration to process
   * @param {Object} sourceNode - Node containing the function
   */
  function processHandler(functionConfig, sourceNode) {
    if (sourceNode.properties.function.handler) {
      const handlerName = sourceNode.properties.function.handler;
      if (!handlerName.startsWith("__function__:")) {
        functionConfig.function.handler = `__function__:${handlerName}`;
      }
    }
  }

  /**
   * Finds all functions connected to a node
   * @param {LGraphNode} node - The node to find functions for
   * @returns {Array<Object>} Array of function configurations
   */
  function findConnectedFunctions(node) {
    const functions = [];

    if (node.outputs && node.outputs[0] && node.outputs[0].links) {
      node.outputs[0].links.forEach((linkId) => {
        const link = graphInstance.links[linkId];
        if (!link) return;

        const targetNode = nodes.find((n) => n.id === link.target_id);
        if (!targetNode) return;

        if (targetNode.constructor.name === "PipecatFunctionNode") {
          // Create base function configuration
          const funcConfig = {
            type: "function",
            name: targetNode.properties.function.name,
            function: { ...targetNode.properties.function },
          };

          processHandler(funcConfig, targetNode);

          // Find where this function connects to (if anywhere)
          const functionTargets = targetNode.outputs[0].links || [];
          if (functionTargets.length > 0) {
            // Look through all connections to find the actual target node
            // (skipping merge nodes)
            for (const targetLinkId of functionTargets) {
              const targetLink = graphInstance.links[targetLinkId];
              if (!targetLink) continue;

              const nextNode = nodes.find((n) => n.id === targetLink.target_id);
              if (!nextNode) continue;

              // If it connects to a merge node, follow through to final target
              if (nextNode.constructor.name === "PipecatMergeNode") {
                const mergeOutput = nextNode.outputs[0].links?.[0];
                if (!mergeOutput) continue;

                const mergeLink = graphInstance.links[mergeOutput];
                if (!mergeLink) continue;

                const finalNode = nodes.find(
                  (n) => n.id === mergeLink.target_id,
                );
                if (finalNode) {
                  funcConfig.function.transition_to = finalNode.title;
                  break; // Use first valid target found
                }
              } else {
                // Direct connection to target node
                funcConfig.function.transition_to = nextNode.title;
                break; // Use first valid target found
              }
            }
          }

          functions.push(funcConfig);
        } else if (targetNode.constructor.name === "PipecatMergeNode") {
          // Find all functions that connect to this merge node
          const connectedFunctions = nodes.filter(
            (n) =>
              n.constructor.name === "PipecatFunctionNode" &&
              n.outputs[0].links?.some((l) => {
                const funcLink = graphInstance.links[l];
                return funcLink && funcLink.target_id === targetNode.id;
              }),
          );

          // Find the final target of the merge node
          const mergeOutput = targetNode.outputs[0].links?.[0];
          if (!mergeOutput) return;

          const mergeLink = graphInstance.links[mergeOutput];
          if (!mergeLink) return;

          const finalNode = nodes.find((n) => n.id === mergeLink.target_id);
          if (!finalNode) return;

          // Add all functions with their transition to the final target
          connectedFunctions.forEach((funcNode) => {
            const funcConfig = {
              type: "function",
              function: {
                ...funcNode.properties.function,
                transition_to: finalNode.title,
              },
            };

            processHandler(funcConfig, funcNode);
            functions.push(funcConfig);
          });
        }
      });
    }

    return functions;
  }

  // Build the flow configuration using the start node's title as initial_node
  const flowConfig = {
    initial_node: startNode.title,
    nodes: {},
  };

  // Process all nodes
  nodes.forEach((node) => {
    if (
      node.constructor.name === "PipecatFunctionNode" ||
      node.constructor.name === "PipecatMergeNode"
    ) {
      return;
    }

    // Create node configuration with new message structure
    const nodeConfig = {
      task_messages: node.properties.task_messages,
      functions: findConnectedFunctions(node),
    };

    // Add role_messages if present
    if (node.properties.role_messages?.length > 0) {
      nodeConfig.role_messages = node.properties.role_messages;
    }

    // Add actions if present
    if (node.properties.pre_actions?.length > 0) {
      nodeConfig.pre_actions = node.properties.pre_actions;
    }
    if (node.properties.post_actions?.length > 0) {
      nodeConfig.post_actions = node.properties.post_actions;
    }

    // Use node.title as the node ID
    flowConfig.nodes[node.title] = nodeConfig;
  });

  return flowConfig;
}
