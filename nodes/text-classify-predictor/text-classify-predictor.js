module.exports = function(RED){
  function autoMLpredictorNode(config){
    const path = require('path')
    const utils = require('../../utils/utils')

    var node = this;
    
    //set configurations
    node.file = __dirname + '/text-classify-predictor.py'
    node.topic = 'predicted'
    node.config = {
      path: path.join(config.modelPath, config.modelName),
      modelPathOrName: config.modelPathOrName,
      orient: config.orient
    }

    utils.run(RED, node, config)
  }
  RED.nodes.registerType("text-classify-predictor", autoMLpredictorNode)
}
