module.exports = function(RED){
	function rFCNode(config){
		const path = require('path')
		const utils = require('../../../utils/utils')

		const node = this;

		//set configurations
		node.file = __dirname +  '/text-classify-light-trainer.py'
		node.config = {
			automl: 'text-classify-light-trainer',
			save: path.join(config.savePath, config.saveName),
			orient: config.orient || 'split',
			epochCount: ~~config.epochCount || 50
		}

		utils.run(RED, node, config)
	}
	RED.nodes.registerType("text-classify-light-trainer", rFCNode);
}
