module.exports = function(RED){
	function rFCNode(config){
		const path = require('path')
		const utils = require('../../utils/utils')
		const node = this;

		//set configurations
		node.file = __dirname + '/text-classify-trainer.py'
		node.config = {
			automl: 'text-classify-trainer',
			save: path.join(config.savePath, config.saveName),
			modelPathOrName: config.modelPathOrName,
			orient: config.orient || 'split',
		}

		utils.run(RED, node, config)
	}
	RED.nodes.registerType("text-classify-trainer", rFCNode);
}
