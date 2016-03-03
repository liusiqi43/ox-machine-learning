require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  self.output:cmul(input):cmul(input:gt(0):double())
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  dzdx = input:gt(0):double():cmul(input)*2
  self.gradInput:cmul(dzdx)
  return self.gradInput
end

