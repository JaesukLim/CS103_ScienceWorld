package scienceworld.tasks.specifictasks

import scienceworld.objects.agent.Agent
import scienceworld.objects.containers.MetalPot
import scienceworld.objects.devices.{Sink, Stove}
import scienceworld.objects.substance.food.{Banana, Flour, Jam, Peanut}
import scienceworld.runtime.pythonapi.PythonInterface
import scienceworld.struct.EnvObject
import scienceworld.tasks.{Task, TaskMakeIsolatedRoom, TaskMaker1, TaskModifier, TaskObject, TaskValueBool, TaskValueStr}
import scienceworld.tasks.goals.{Goal, GoalSequence}
import scienceworld.tasks.goals.specificgoals.{GoalFind, GoalInRoomWithObject, GoalMoveToNewLocation, GoalObjectsInSingleContainer}

import scala.collection.mutable.ArrayBuffer

class TaskRecipePipeline(val mode:String = TaskRecipePipeline.MODE_RECIPE_TINY) extends TaskParametric {
  import TaskRecipePipeline._

  override val taskName:String = mode.replaceAll(" ", "-").replaceAll("[()]", "")
  override val isVisibleInTaskList:Boolean = mode != TaskRecipePipeline.MODE_RECIPE_UNSEEN

  private val miniRoomName = "test kitchen"
  private val seenFlourRooms = Array("living room", "bedroom", "art studio")
  private val unseenFlourRooms = Array("living room", "bedroom")
  private val seenFinishingRooms = Array("art studio", "workshop")
  private val unseenFinishingRooms = Array("art studio", "workshop")

  private val tinyRecipes = Array(
    RecipeSpec("bread", Array.empty[String]),
    RecipeSpec("jam sandwich", Array("jam")),
    RecipeSpec("peanut butter sandwich", Array("peanut"))
  )

  private val seenRecipes = Array(
    RecipeSpec("jam sandwich", Array("jam")),
    RecipeSpec("peanut butter sandwich", Array("peanut")),
    RecipeSpec("banana sandwich", Array("banana"))
  )

  private val unseenRecipes = Array(
    RecipeSpec("peanut butter with jam sandwich", Array("peanut", "jam")),
    RecipeSpec("peanut butter with banana sandwich", Array("peanut", "banana"))
  )

  private val combinations = new ArrayBuffer[Array[TaskModifier]]()

  if (mode == TaskRecipePipeline.MODE_RECIPE_TINY) {
    for (recipe <- tinyRecipes) {
      combinations.append(makeRecipeModifiers(
        recipe = recipe,
        flourRoom = miniRoomName,
        finishingRooms = Array.fill(recipe.finishingIngredients.length)(miniRoomName),
        heatRoom = miniRoomName,
        isMini = true
      ))
    }
  } else if (mode == TaskRecipePipeline.MODE_RECIPE_SEEN) {
    for (recipe <- seenRecipes; flourRoom <- seenFlourRooms) {
      combinations.append(makeRecipeModifiers(
        recipe = recipe,
        flourRoom = flourRoom,
        finishingRooms = pickFinishingRooms(recipe.finishingIngredients.length, flourRoom, seenFinishingRooms),
        heatRoom = "kitchen",
        isMini = false
      ))
    }
  } else {
    for (recipe <- unseenRecipes; flourRoom <- unseenFlourRooms) {
      combinations.append(makeRecipeModifiers(
        recipe = recipe,
        flourRoom = flourRoom,
        finishingRooms = pickFinishingRooms(recipe.finishingIngredients.length, flourRoom, unseenFinishingRooms),
        heatRoom = "kitchen",
        isMini = false
      ))
    }
  }

  override def numCombinations(): Int = combinations.length

  override def getCombination(idx:Int): Array[TaskModifier] = combinations(idx)

  private def makeRecipeModifiers(
    recipe:RecipeSpec,
    flourRoom:String,
    finishingRooms:Array[String],
    heatRoom:String,
    isMini:Boolean
  ): Array[TaskModifier] = {
    val modifiers = new ArrayBuffer[TaskModifier]()

    if (isMini) {
      modifiers.append(new TaskMakeIsolatedRoom(miniRoomName))
      modifiers.append(new TaskObject("sink", Some(new Sink(None)), miniRoomName, Array.empty[String], forceAdd = true))
      modifiers.append(new TaskObject("stove", Some(new Stove()), miniRoomName, Array.empty[String], forceAdd = true))
    } else {
      modifiers.append(new TaskObject("sink", Some(new Sink(None)), heatRoom, Array.empty[String]))
      modifiers.append(new TaskObject("stove", Some(new Stove()), heatRoom, Array.empty[String]))
    }

    modifiers.append(new TaskObject("metal pot", Some(new MetalPot()), heatRoom, Array.empty[String], forceAdd = true))
    modifiers.append(new TaskObject("flour", Some(new Flour()), flourRoom, Array.empty[String], forceAdd = true))

    for ((ingredientName, roomName) <- recipe.finishingIngredients.zip(finishingRooms)) {
      modifiers.append(new TaskObject(ingredientName, Some(mkIngredientObject(ingredientName)), roomName, Array.empty[String], forceAdd = true))
    }

    modifiers.append(new TaskValueStr("targetName", recipe.targetName))
    modifiers.append(new TaskValueStr("flourRoom", flourRoom))
    modifiers.append(new TaskValueStr("heatRoom", heatRoom))
    modifiers.append(new TaskValueStr("finishingIngredients", recipe.finishingIngredients.mkString(",")))
    modifiers.append(new TaskValueStr("finishingRooms", finishingRooms.mkString(",")))
    modifiers.append(new TaskValueBool("isMini", isMini))
    modifiers.append(new TaskValueStr("recipe_steps", mkRecipeSteps(recipe, flourRoom, finishingRooms, heatRoom).mkString("\n")))

    modifiers.toArray
  }

  private def pickFinishingRooms(numIngredients:Int, flourRoom:String, candidateRooms:Array[String]): Array[String] = {
    if (numIngredients == 0) return Array.empty[String]

    val usableRooms = candidateRooms.filterNot(_ == flourRoom)
    if (usableRooms.isEmpty) {
      return Array.fill(numIngredients)(flourRoom)
    }

    usableRooms.take(numIngredients)
  }

  private def mkIngredientObject(name:String): EnvObject = {
    name match {
      case "jam" => new Jam()
      case "peanut" => new Peanut()
      case "banana" => new Banana()
      case _ => new Jam()
    }
  }

  private def mkRecipeSteps(
    recipe:RecipeSpec,
    flourRoom:String,
    finishingRooms:Array[String],
    heatRoom:String
  ): Array[String] = {
    val steps = new ArrayBuffer[String]()

    steps.append("Target dish: " + recipe.targetName + ".")
    steps.append("Pick up the metal pot in " + heatRoom + ".")
    steps.append("Use the sink in " + heatRoom + " to fill the metal pot with water.")
    steps.append("Put the flour from " + flourRoom + " into the metal pot, then mix the metal pot to make dough.")
    steps.append("Heat the dough on the stove in " + heatRoom + " until it becomes bread.")

    if (recipe.finishingIngredients.nonEmpty) {
      val finishingStr = recipe.finishingIngredients.zip(finishingRooms).map {
        case (ingredientName, roomName) => "the " + ingredientName + " from " + roomName
      }.mkString(" and ")
      steps.append("Add " + finishingStr + " to the bread in the metal pot, then mix the metal pot again to make " + recipe.targetName + ".")
    }

    steps.append("Focus on the " + recipe.targetName + ".")
    steps.toArray
  }

  private def parseCSV(value:String): Array[String] = {
    if (value.trim.isEmpty) Array.empty[String]
    else value.split(",").map(_.trim).filter(_.nonEmpty)
  }

  private def runModifiers(modifiers:Array[TaskModifier], universe:EnvObject, agent:Agent):(Boolean, String) = {
    for (mod <- modifiers) {
      val success = mod.runModifier(universe, agent)
      if (!success) return (false, "ERROR: Error running one or more modifiers while setting up task environment.")
    }
    (true, "")
  }

  override def setupCombination(combinationNum:Int, universe:EnvObject, agent:Agent): (Boolean, String) = {
    if (combinationNum >= this.numCombinations()) {
      return (false, "ERROR: The requested variation (" + combinationNum + ") exceeds the total number of variations (" + this.numCombinations() + ").")
    }
    this.runModifiers(this.getCombination(combinationNum), universe, agent)
  }

  private def mkDescription(isMini:Boolean): String = {
    if (isMini) {
      "Your task is to follow the recipe, prepare the final dish, and focus on it. Everything you need is in this room."
    } else if (mode == TaskRecipePipeline.MODE_RECIPE_SEEN) {
      "Your task is to follow the recipe, gather ingredients from around the house, use the kitchen to cook the dish, and then focus on the final result."
    } else {
      "Your task is to follow the recipe, search for ingredients, use the kitchen to cook the dish, and then focus on the final result."
    }
  }

  private def setupGoals(modifiers:Array[TaskModifier], combinationNum:Int): Task = {
    val targetName = this.getTaskValueStr(modifiers, "targetName").get
    val flourRoom = this.getTaskValueStr(modifiers, "flourRoom").get
    val finishingIngredients = parseCSV(this.getTaskValueStr(modifiers, "finishingIngredients").getOrElse(""))
    val heatRoom = this.getTaskValueStr(modifiers, "heatRoom").get
    val isMini = this.getTaskValueBool(modifiers, "isMini").getOrElse(false)

    val gSequence = new ArrayBuffer[Goal]
    gSequence.append(new GoalFind(objectName = targetName, failIfWrong = true, description = "focus on the " + targetName))

    val gSequenceUnordered = new ArrayBuffer[Goal]
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "metal pot", _isOptional = true, description = "find the metal pot"))
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "flour", _isOptional = true, description = "find the flour"))
    gSequenceUnordered.append(new GoalObjectsInSingleContainer(objectNames = Array("flour", "water"), _isOptional = true, description = "put flour and water together in one container"))
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "dough", _isOptional = true, description = "make dough"))
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "bread", _isOptional = true, description = "make bread"))
    if (finishingIngredients.nonEmpty) {
      gSequenceUnordered.append(new GoalObjectsInSingleContainer(objectNames = Array("bread") ++ finishingIngredients, _isOptional = true, description = "combine bread with the finishing ingredients"))
    }
    for (ingredientName <- finishingIngredients) {
      gSequenceUnordered.append(new GoalInRoomWithObject(objectName = ingredientName, _isOptional = true, description = "find the " + ingredientName))
    }
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "stove", _isOptional = true, description = "use the stove in " + heatRoom))
    if (!isMini && flourRoom != heatRoom) {
      gSequenceUnordered.append(new GoalMoveToNewLocation(_isOptional = true, description = "move to a new location"))
    }

    val description = mkDescription(isMini)
    new Task(taskName, description, new GoalSequence(gSequence.toArray, gSequenceUnordered.toArray), taskModifiers = modifiers)
  }

  override def setupGoals(combinationNum:Int): Task = this.setupGoals(this.getCombination(combinationNum), combinationNum)

  override def mkGoldActionSequence(modifiers:Array[TaskModifier], runner:PythonInterface): (Boolean, Array[String]) = {
    (true, Array.empty[String])
  }
}

object TaskRecipePipeline {
  val MODE_RECIPE_TINY = "recipe pipeline tiny"
  val MODE_RECIPE_SEEN = "recipe pipeline seen"
  val MODE_RECIPE_UNSEEN = "recipe pipeline unseen"

  case class RecipeSpec(targetName:String, finishingIngredients:Array[String])

  def registerTasks(taskMaker:TaskMaker1): Unit = {
    taskMaker.addTask(new TaskRecipePipeline(mode = MODE_RECIPE_TINY))
    taskMaker.addTask(new TaskRecipePipeline(mode = MODE_RECIPE_SEEN))
    taskMaker.addTask(new TaskRecipePipeline(mode = MODE_RECIPE_UNSEEN))
  }
}
