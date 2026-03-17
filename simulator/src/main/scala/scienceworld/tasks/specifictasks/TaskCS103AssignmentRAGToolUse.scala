package scienceworld.tasks.specifictasks

import scienceworld.objects.agent.Agent
import scienceworld.objects.substance.food.{Almond, Apple, Banana, Bread, Cashew, Jam, Orange, Peanut}
import scienceworld.runtime.pythonapi.PythonInterface
import scienceworld.struct.EnvObject
import scienceworld.tasks.goals.{Goal, GoalSequence}
import scienceworld.tasks.goals.specificgoals.{GoalFind, GoalInRoomWithObject, GoalMoveToNewLocation, GoalObjectsInSingleContainer, GoalPastActionReadObject}
import scienceworld.tasks.{Task, TaskMaker1, TaskModifier}

import scala.collection.mutable.ArrayBuffer

class TaskCS103AssignmentRAGToolUse(
  val mode:String = TaskCS103AssignmentRAGToolUse.MODE_ASSIGNMENT_2_RAG_TOOL_USE
) extends TaskParametric {
  val taskName:String = mode.replaceAll(" ", "-").replaceAll("[()]", "")

  private val combinations = new ArrayBuffer[Array[TaskModifier]]()
  private val chemistryMixHelper = new TaskChemistryMix(TaskChemistryMix.MODE_CHEMISTRY_MIX)

  combinations.append(
    TaskChemistryMix.setupRecipeTask(
      resultObject = "peanut butter sandwich",
      inputObjects = Array(new Peanut(), new Bread()),
      generateLocation = "workshop"
    )
  )
  combinations.append(
    TaskChemistryMix.setupRecipeTask(
      resultObject = "jam sandwich",
      inputObjects = Array(new Jam(), new Bread()),
      generateLocation = "workshop"
    )
  )
  combinations.append(
    TaskChemistryMix.setupRecipeTask(
      resultObject = "banana sandwich",
      inputObjects = Array(new Banana(), new Bread()),
      generateLocation = "workshop"
    )
  )
  combinations.append(
    TaskChemistryMix.setupRecipeTask(
      resultObject = "mixed nuts",
      inputObjects = Array(new Peanut(), new Almond(), new Cashew()),
      generateLocation = "workshop"
    )
  )
  combinations.append(
    TaskChemistryMix.setupRecipeTask(
      resultObject = "fruit salad",
      inputObjects = Array(new Apple(), new Orange(), new Banana()),
      generateLocation = "workshop"
    )
  )

  override def numCombinations(): Int = combinations.length

  override def getCombination(idx:Int): Array[TaskModifier] = combinations(idx)

  private def setupCombination(modifiers:Array[TaskModifier], universe:EnvObject, agent:Agent):(Boolean, String) = {
    for (mod <- modifiers) {
      val success = mod.runModifier(universe, agent)
      if (!success) {
        return (false, "ERROR: Failed to apply one or more assignment RAG/tool-use modifiers.")
      }
    }
    (true, "")
  }

  override def setupCombination(combinationNum:Int, universe:EnvObject, agent:Agent):(Boolean, String) = {
    if (combinationNum >= this.numCombinations()) {
      return (
        false,
        "ERROR: The requested variation (" + combinationNum + ") exceeds the total number of variations (" + this.numCombinations() + ")."
      )
    }
    this.setupCombination(this.getCombination(combinationNum), universe, agent)
  }

  private def setupGoals(modifiers:Array[TaskModifier], combinationNum:Int): Task = {
    val resultChemical = this.getTaskValueStr(modifiers, key = "result").get
    val inputChemicals = this.getTaskValueStr(modifiers, key = "inputChemicals").get.split(",")
    val location = this.getTaskValueStr(modifiers, key = "location").get

    val gSequence = new ArrayBuffer[Goal]
    val gSequenceUnordered = new ArrayBuffer[Goal]

    gSequence.append(new GoalFind(objectName = resultChemical, failIfWrong = true, description = "focus on the final mixture"))

    for (inputChemical <- inputChemicals) {
      gSequenceUnordered.append(
        new GoalInRoomWithObject(
          objectName = inputChemical,
          _isOptional = true,
          description = "observe ingredient " + inputChemical
        )
      )
    }
    gSequenceUnordered.append(
      new GoalObjectsInSingleContainer(
        objectNames = inputChemicals,
        _isOptional = true,
        description = "place all ingredients in one container"
      )
    )
    gSequenceUnordered.append(
      new GoalPastActionReadObject(
        documentName = "recipe",
        _isOptional = true,
        description = "read the recipe"
      )
    )
    gSequenceUnordered.append(new GoalMoveToNewLocation(_isOptional = true, description = "move away from the starting location"))

    var description = "This is CS103 Assignment 2 (RAG and Tool Use). "
    description += "A recipe and the ingredients are in the " + location + ". "
    description += "Read the recipe, use a container from the kitchen, mix the required ingredients, "
    description += "and then focus on the " + resultChemical + "."

    new Task(taskName, description, new GoalSequence(gSequence.toArray, gSequenceUnordered.toArray), taskModifiers = modifiers)
  }

  override def setupGoals(combinationNum:Int): Task = {
    this.setupGoals(this.getCombination(combinationNum), combinationNum)
  }

  override def mkGoldActionSequence(modifiers:Array[TaskModifier], runner:PythonInterface):(Boolean, Array[String]) = {
    chemistryMixHelper.mkGoldActionSequence(modifiers, runner)
  }
}

object TaskCS103AssignmentRAGToolUse {
  val MODE_ASSIGNMENT_2_RAG_TOOL_USE = "assignment 2 rag tool use"

  def registerTasks(taskMaker:TaskMaker1): Unit = {
    taskMaker.addTask(new TaskCS103AssignmentRAGToolUse())
  }
}
