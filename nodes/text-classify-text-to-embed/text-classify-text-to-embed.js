module.exports = function(RED){
	function rFCNode(config){
		const utils = require('../../utils/utils')

		const node = this;

		//set configurations
		node.file = __dirname +  '/text-classify-text-to-embed.py'
		node.config = {
			automl: 'text-classify-text-to-embed',
			modelPathOrName: config.modelPathOrName,
			orient: config.orient || 'split',
		}

		utils.run(RED, node, config)
	}
	RED.nodes.registerType("text-classify-text-to-embed", rFCNode);
}
