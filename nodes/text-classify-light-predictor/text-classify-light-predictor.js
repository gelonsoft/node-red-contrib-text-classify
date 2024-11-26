module.exports = function(RED){
	function rFCNode(config){
		const path = require('path')
		const utils = require('../../../utils/utils')

		const node = this;

		//set configurations
		node.file = __dirname +  '/text-classify-light-predictor.py'
		node.config = {
			automl: 'text-classify-light-predictor',
			save: path.join(config.savePath, config.saveName),
			tokenizerPathOrName: config.tokenizerPathOrName,
			modelPathOrName: config.modelPathOrName,
			orient: config.orient || 'split',
		}

		utils.run(RED, node, config)
	}
	RED.nodes.registerType("text-classify-light-predictor", rFCNode);
}
